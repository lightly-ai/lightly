""" Contrastive Loss Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved


from typing import Sequence, Union

from torch import Tensor

from lightly.loss import NTXentLoss


class DirectCLRLoss(NTXentLoss):
    """Implementation of the NT-Xent based DirectCLR Loss.

    Following the DirectCLR[0] paper, this loss should be used without projection
    head. Set `loss_dim` to the desired truncated representation length.
    DirectCLRLoss inherits from NTXentLoss, its parameters can be set after
    setting `loss_dim`.

    - [0] DirectCLR, 2021, https://arxiv.org/abs/2110.09348

    Attributes:
        loss_dim:
            Computes the loss only on the first loss_dim values of the encoding.
        temperature:
            From NTXentLoss: scale logits by the inverse of the temperature.
        memory_bank_size:
            From NTXentLoss: size of the memory bank as (num_features, dim) tuple.
            num_features are the number of negative samples stored in the memory bank.
            If num_features is 0, the memory bank is disabled. Use 0 for SimCLR. For
            MoCo we typically use numbers like 4096 or 65536.
            Deprecated: If only a single integer is passed, it is interpreted as the
            number of features and the feature dimension is inferred from the first
            batch stored in the memory bank. Leaving out the feature dimension might
            lead to errors in distributed training.
        gather_distributed:
            From NTXentLoss: if True then negatives from all GPUs are gathered before
            the loss calculation. If a memory bank is used and gather_distributed is
            True, then tensors from all gpus are gathered before the memory bank is
            updated.

    Examples:
        >>> # initialize loss function
        >>> loss_fn = DirectCLRLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through backbone without projection head
        >>> out0, out1 = model(t0), model(t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)

    """

    def __init__(
        self,
        loss_dim: int = 64,
        temperature: float = 0.5,
        memory_bank_size: Union[int, Sequence[int]] = 0,
        gather_distributed: bool = False,
    ):
        """Initializes the DirectCLRLoss module with the specified parameters.

        Args:
            loss_dim:
                Computes the loss only on the first `loss_dim` values of the encoding.
            temperature:
                 Scale logits by the inverse of the temperature.
            memory_bank_size:
                 Size of the memory bank.
            gather_distributed:
                 If True, negatives from all GPUs are gathered before the loss calculation.
        """
        super().__init__(
            temperature=temperature,
            memory_bank_size=memory_bank_size,
            gather_distributed=gather_distributed,
        )
        self.loss_dim = loss_dim

    def forward(self, out0: Tensor, out1: Tensor) -> Tensor:
        """Forward pass through DirectCLR Loss.

        To be used directly on the encoding without projection head. Flattens
        each output encoding and truncates it to `loss_dim` length, then computes
        the NTXentLoss.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)

        Returns:
            DirectCLR Loss value.
        """

        out0 = out0.flatten(start_dim=1)[:, : self.loss_dim]
        out1 = out1.flatten(start_dim=1)[:, : self.loss_dim]

        loss: Tensor = super().forward(out0, out1)

        return loss
