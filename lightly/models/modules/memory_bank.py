""" Memory Bank Wrapper """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from lightly.models import utils
from lightly.utils import dist


class MemoryBankModule(Module):
    """Memory bank implementation

    This is a parent class to all loss functions implemented by the lightly
    Python package. This way, any loss can be used with a memory bank if
    desired.

    Attributes:
        size:
            Size of the memory bank as (num_features, dim) tuple. If num_features is 0
            then the memory bank is disabled. Deprecated: If only a single integer is
            passed, it is interpreted as the number of features and the feature
            dimension is inferred from the first batch stored in the memory bank.
            Leaving out the feature dimension might lead to errors in distributed
            training.
        gather_distributed:
            If True then negatives from all gpus are gathered before the memory bank
            is updated. This results in more frequent updates of the memory bank and
            keeps the memory bank contents independent of the number of gpus. But it has
            the drawback that synchronization between processes is required and
            diversity of the memory bank content is reduced.
        feature_dim_first:
            If True, the memory bank returns features with shape (dim, num_features).
            If False, the memory bank returns features with shape (num_features, dim).

    Examples:
        >>> class MyLossFunction(MemoryBankModule):
        >>>
        >>>     def __init__(self, memory_bank_size: Tuple[int, int] = (2 ** 16, 128)):
        >>>         super().__init__(memory_bank_size)
        >>>
        >>>     def forward(self, output: Tensor, labels: Optional[Tensor] = None):
        >>>         output, negatives = super().forward(output)
        >>>
        >>>         if negatives is not None:
        >>>             # evaluate loss with negative samples
        >>>         else:
        >>>             # evaluate loss without negative samples

    """

    def __init__(
        self,
        size: Union[int, Sequence[int]] = 65536,
        gather_distributed: bool = False,
        feature_dim_first: bool = True,
    ):
        super().__init__()
        size_tuple = (size,) if isinstance(size, int) else tuple(size)

        if any(x < 0 for x in size_tuple):
            raise ValueError(
                f"Illegal memory bank size {size}, all entries must be non-negative."
            )

        self.size = size_tuple
        self.gather_distributed = gather_distributed
        self.feature_dim_first = feature_dim_first
        self.bank: Tensor
        self.register_buffer(
            "bank",
            tensor=torch.empty(size=size_tuple, dtype=torch.float),
            persistent=False,
        )
        self.bank_ptr: Tensor
        self.register_buffer(
            "bank_ptr",
            tensor=torch.empty(1, dtype=torch.long),
            persistent=False,
        )

        if isinstance(size, int) and size > 0:
            warnings.warn(
                (
                    f"Memory bank size 'size={size}' does not specify feature "
                    "dimension. It is recommended to set the feature dimension with "
                    "'size=(n, dim)' when creating the memory bank. Distributed "
                    "training might fail if the feature dimension is not set."
                ),
                UserWarning,
            )
        elif len(size_tuple) > 1:
            self._init_memory_bank(size=size_tuple)

    def forward(
        self,
        output: Tensor,
        labels: Optional[Tensor] = None,
        update: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, None]]:
        """Query memory bank for additional negative samples

        Args:
            output:
                The output of the model.
            labels:
                Should always be None, will be ignored.
            update:
                If True, the memory bank will be updated with the current output.

        Returns:
            The output if the memory bank is of size 0, otherwise the output
            and the entries from the memory bank. Entries from the memory bank have
            shape (dim, num_features) if feature_dim_first is True and
            (num_features, dim) otherwise.

        """

        # no memory bank, return the output
        if self.size[0] == 0:
            return output, None

        # Initialize the memory bank if it is not already done.
        if self.bank.ndim == 1:
            dim = output.shape[1:]
            self._init_memory_bank(size=(*self.size, *dim))

        # query and update memory bank
        bank = self.bank.clone().detach()
        if self.feature_dim_first:
            # swap bank size and feature dimension for backwards compatibility
            bank = bank.transpose(0, -1)

        # only update memory bank if we later do backward pass (gradient)
        if update:
            self._dequeue_and_enqueue(output)

        return output, bank

    @torch.no_grad()
    def _init_memory_bank(self, size: Tuple[int, ...]) -> None:
        """Initialize the memory bank.

        Args:
            size:
                Size of the memory bank as (num_features, dim) tuple.

        """
        self.bank = torch.randn(size).type_as(self.bank)
        self.bank = torch.nn.functional.normalize(self.bank, dim=-1)
        self.bank_ptr = torch.zeros(1).type_as(self.bank_ptr)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: Tensor) -> None:
        """Dequeue the oldest batch and add the latest one

        Args:
            batch:
                The latest batch of keys to add to the memory bank.

        """
        if self.gather_distributed and dist.world_size() > 1:
            batch = utils.concat_all_gather(batch)

        batch_size = batch.shape[0]
        ptr = int(self.bank_ptr)
        if ptr + batch_size >= self.size[0]:
            self.bank[ptr:] = batch[: self.size[0] - ptr].detach()
            self.bank_ptr.zero_()
        else:
            self.bank[ptr : ptr + batch_size] = batch.detach()
            self.bank_ptr[0] = ptr + batch_size
