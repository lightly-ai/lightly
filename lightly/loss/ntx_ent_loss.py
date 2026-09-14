"""Contrastive Loss Functions"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import Sequence, Union

import torch
from torch import Tensor, nn
from torch import distributed as torch_dist

from lightly.functional.ntxent import ntxent
from lightly.loss.distributed import DistributedKind
from lightly.models.modules.memory_bank import MemoryBankModule
from lightly.utils import dist


class NTXentLoss(nn.Module):
    """Implementation of the Contrastive Cross Entropy Loss.

    This implementation follows the SimCLR[0] paper. If you enable the memory
    bank by setting the `memory_bank_size` value > 0 the loss behaves like
    the one described in the MoCo[1] paper.

    - [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709
    - [1] MoCo, 2020, https://arxiv.org/abs/1911.05722

    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        memory_bank_size:
            Size of the memory bank as (num_features, dim) tuple. num_features are the
            number of negative samples stored in the memory bank. If num_features is 0,
            the memory bank is disabled. Use 0 for SimCLR. For MoCo we typically use
            numbers like 4096 or 65536.
            Deprecated: If only a single integer is passed, it is interpreted as the
            number of features and the feature dimension is inferred from the first
            batch stored in the memory bank. Leaving out the feature dimension might
            lead to errors in distributed training.
        gather_distributed:
            If True then negatives from all GPUs are gathered before the
            loss calculation. If a memory bank is used and gather_distributed is True,
            then tensors from all gpus are gathered before the memory bank is updated.

    Raises:
        ValueError: If abs(temperature) < 1e-8 to prevent divide by zero.

    Examples:
        >>> # initialize loss function without memory bank
        >>> loss_fn = NTXentLoss(memory_bank_size=0)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimCLR or MoCo model
        >>> out0, out1 = model(t0), model(t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)

    """

    distributed_kind = DistributedKind.GATHER_FOR_NEGATIVES

    def __init__(
        self,
        temperature: float = 0.5,
        memory_bank_size: Union[int, Sequence[int]] = 0,
        gather_distributed: bool = False,
    ):
        """Initializes the NTXentLoss module with the specified parameters.

        Args:
            temperature:
                 Scale logits by the inverse of the temperature.
            memory_bank_size:
                 Size of the memory bank.
            gather_distributed:
                 If True, negatives from all GPUs are gathered before the loss calculation.

        Raises:
            ValueError: If temperature is less than 1e-8 to prevent divide by zero.
            ValueError: If gather_distributed is True but torch.distributed is not available.
        """
        super().__init__()
        self.memory_bank = MemoryBankModule(
            size=memory_bank_size, gather_distributed=gather_distributed
        )
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.eps = 1e-8

        if abs(self.temperature) < self.eps:
            raise ValueError(
                "Illegal temperature: abs({}) < 1e-8".format(self.temperature)
            )
        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(self, out0: Tensor, out1: Tensor) -> Tensor:
        """Forward pass through Contrastive Cross-Entropy Loss.

        If used with a memory bank, the samples from the memory bank are used
        as negative examples. Otherwise, within-batch samples are used as
        negative samples.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)

        Returns:
            Contrastive Cross Entropy Loss value.
        """
        # Normalize the output to length 1. The memory bank stores unit vectors,
        # and ntxent takes them, so this happens here and only here.
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        # ask memory bank for negative samples and extend it with out1 if
        # out1 requires a gradient, otherwise keep the same vectors in the
        # memory bank (this allows for keeping the memory bank constant e.g.
        # for evaluating the loss on the test set)
        # out1: shape: (batch_size, embedding_size)
        # negatives: shape: (embedding_size, memory_bank_size)
        out1, negatives = self.memory_bank.forward(out1, update=out0.requires_grad)

        if negatives is not None:
            return ntxent(out0, out1, temperature=self.temperature, negatives=negatives)

        if self.gather_distributed and dist.world_size() > 1:
            # GATHER_FOR_NEGATIVES: every sample on every rank is a negative for
            # every other, so the gather happens with gradient before the loss.
            return ntxent(
                out0,
                out1,
                temperature=self.temperature,
                z0_all=torch.cat(dist.gather(out0), 0),
                z1_all=torch.cat(dist.gather(out1), 0),
                positives_at=dist.rank() * out0.size(0),
            )

        return ntxent(out0, out1, temperature=self.temperature)
