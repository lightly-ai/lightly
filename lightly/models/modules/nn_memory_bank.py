""" Nearest Neighbour Memory Bank Module """

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

from typing import Optional

import torch
from torch import Tensor

from lightly.loss.memory_bank import MemoryBankModule


class NNMemoryBankModule(MemoryBankModule):
    """Nearest Neighbour Memory Bank implementation

    This class implements a nearest neighbour memory bank as described in the
    NNCLR paper[0]. During the forward pass we return the nearest neighbour
    from the memory bank.

    [0] NNCLR, 2021, https://arxiv.org/abs/2104.14548

    Attributes:
        size:
            Number of keys the memory bank can store.

    Examples:
        >>> model = NNCLR(backbone)
        >>> criterion = NTXentLoss(temperature=0.1)
        >>>
        >>> nn_replacer = NNmemoryBankModule(size=2 ** 16)
        >>>
        >>> # forward pass
        >>> (z0, p0), (z1, p1) = model(x0, x1)
        >>> z0 = nn_replacer(z0.detach(), update=False)
        >>> z1 = nn_replacer(z1.detach(), update=True)
        >>>
        >>> loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))

    """

    def __init__(self, size: int = 2**16):
        if size <= 0:
            raise ValueError(f"Memory bank size must be positive, got {size}.")
        super(NNMemoryBankModule, self).__init__(size)

    def forward(  # type: ignore[override] # TODO(Philipp, 11/23): Fix signature to match parent class.
        self,
        output: Tensor,
        update: bool = False,
    ) -> Tensor:
        """Returns nearest neighbour of output tensor from memory bank

        Args:
            output: The torch tensor for which you want the nearest neighbour
            update: If `True` updated the memory bank by adding output to it

        """

        output, bank = super(NNMemoryBankModule, self).forward(output, update=update)
        assert bank is not None
        bank = bank.to(output.device).t()

        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)

        similarity_matrix = torch.einsum("nd,md->nm", output_normed, bank_normed)
        index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
        nearest_neighbours = torch.index_select(
            bank, dim=0, index=index_nearest_neighbours
        )

        return nearest_neighbours
