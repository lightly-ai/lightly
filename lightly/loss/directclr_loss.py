""" Contrastive Loss Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved


from typing import Any

from torch import Tensor

from lightly.loss import NTXentLoss


class DirectCLRLoss(NTXentLoss):
    """Implementation of the NT-Xent based DirectCLR Loss.

    Following the DirectCRL[0] paper, this loss should be used without projection
    head. Set loss_dim to the desired truncated representation length.
    DirectCLRLoss inherits from NTXentLoss, its parameters can be set after
    setting loss_dim.

    - [0] DirectCLR, 2021, https://arxiv.org/abs/2110.09348

    Attributes:
        loss_dim:
            Computes the loss only on the first loss_dim values of the encoding.
        *args:
            Positional arguments for NTXentLoss.
        **kwargs:
            Keyword arguments for NTXentLoss.

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
        *args: Any,
        **kwargs: Any,
    ):
        """Initializes the DirectCLRLoss module with the specified parameters.

        Args:
            loss_dim:
                Computes the loss only on the first loss_dim values of the encoding.
            *args:
                Positional arguments for NTXentLoss.
            **kwargs:
                Keyword arguments for NTXentLoss.
        """
        super().__init__(*args, **kwargs)
        self.loss_dim = loss_dim

    def forward(self, out0: Tensor, out1: Tensor) -> Tensor:
        """Forward pass through DirectCLR Loss.

        To be used directly on the encoding without projection head. Flattens
        each output encoding and truncates it to loss_dim length, then computes
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
