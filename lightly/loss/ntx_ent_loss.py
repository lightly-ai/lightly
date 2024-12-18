""" Contrastive Loss Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import Sequence, Union

import torch
from torch import Tensor
from torch import distributed as torch_dist
from torch import nn

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
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(output)

    """

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
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
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

        device = out0.device
        batch_size, _ = out0.shape

        # Normalize the output to length 1
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        # ask memory bank for negative samples and extend it with out1 if
        # out1 requires a gradient, otherwise keep the same vectors in the
        # memory bank (this allows for keeping the memory bank constant e.g.
        # for evaluating the loss on the test set)
        # out1: shape: (batch_size, embedding_size)
        # negatives: shape: (embedding_size, memory_bank_size)
        out1, negatives = self.memory_bank.forward(out1, update=out0.requires_grad)

        # Use cosine similarity (dot product) as all vectors are normalized to unit length
        # Notation in einsum: n = batch_size, c = embedding_size and k = memory_bank_size.

        if negatives is not None:
            # Use negatives from memory bank
            negatives = negatives.to(device)

            # sim_pos is of shape (batch_size, 1) and sim_pos[i] denotes the similarity
            # of the i-th sample in the batch to its positive pair
            sim_pos = torch.einsum("nc,nc->n", out0, out1).unsqueeze(-1)

            # sim_neg is of shape (batch_size, memory_bank_size) and sim_neg[i,j] denotes the similarity
            # of the i-th sample to the j-th negative sample
            sim_neg = torch.einsum("nc,ck->nk", out0, negatives)

            # Set the labels to maximize sim_pos in relation to sim_neg
            logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
            labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)

        else:
            # Use other samples from batch as negatives
            # and create diagonal mask that only selects similarities between
            # views of the same image
            if self.gather_distributed and dist.world_size() > 1:
                # Gather hidden representations from other processes
                out0_large = torch.cat(dist.gather(out0), 0)
                out1_large = torch.cat(dist.gather(out1), 0)
                diag_mask = dist.eye_rank(batch_size, device=out0.device)
            else:
                # Single process
                out0_large = out0
                out1_large = out1
                diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)

            # Calculate similiarities
            # Here n = batch_size and m = batch_size * world_size
            # The resulting vectors have shape (n, m)
            logits_00 = torch.einsum("nc,mc->nm", out0, out0_large) / self.temperature
            logits_01 = torch.einsum("nc,mc->nm", out0, out1_large) / self.temperature
            logits_10 = torch.einsum("nc,mc->nm", out1, out0_large) / self.temperature
            logits_11 = torch.einsum("nc,mc->nm", out1, out1_large) / self.temperature

            # Remove simliarities between same views of the same image
            logits_00 = logits_00[~diag_mask].view(batch_size, -1)
            logits_11 = logits_11[~diag_mask].view(batch_size, -1)

            # Concatenate logits
            # The logits tensor in the end has shape (2*n, 2*m-1)
            logits_0100 = torch.cat([logits_01, logits_00], dim=1)
            logits_1011 = torch.cat([logits_10, logits_11], dim=1)
            logits = torch.cat([logits_0100, logits_1011], dim=0)

            # Create labels
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            if self.gather_distributed:
                labels = labels + dist.rank() * batch_size
            labels = labels.repeat(2)

        # Calculate the cross-entropy loss
        loss: Tensor = self.cross_entropy(logits, labels)

        return loss
