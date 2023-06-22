""" Memory Bank Wrapper """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import warnings
from typing import Sequence, Union

import torch
from torch import Tensor
from torch.nn import Module

from lightly.models import utils


class MemoryBankModule(Module):
    """Memory bank implementation

    This is a parent class to all loss functions implemented by the lightly
    Python package. This way, any loss can be used with a memory bank if
    desired.

    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.
        gather_distributed:
            If True then negatives from all gpus are gathered before the memory bank
            is updated. This results in more frequent updates of the memory bank and
            keeps the memory bank contents independent of the number of gpus. But it has
            the drawback that synchronization between processes is required and
            diversity of the memory bank content is reduced.

    Examples:
        >>> class MyLossFunction(MemoryBankModule):
        >>>
        >>>     def __init__(self, memory_bank_size: int = 2 ** 16):
        >>>         super(MyLossFunction, self).__init__(memory_bank_size)
        >>>
        >>>     def forward(self, output: Tensor,
        >>>                 labels: Tensor = None):
        >>>
        >>>         output, negatives = super(
        >>>             MyLossFunction, self).forward(output)
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
    ):
        super(MemoryBankModule, self).__init__()
        size_tuple = (size,) if isinstance(size, int) else tuple(size)

        if any(x < 0 for x in size_tuple):
            raise ValueError(
                f"Illegal memory bank size {size}, all entries must be non-negative."
            )

        self.size = size_tuple
        self.gather_distributed = gather_distributed
        self.register_buffer(
            "bank",
            tensor=torch.empty(size=self.size, dtype=torch.float),
            persistent=False,
        )
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
        else:
            self._init_memory_bank(dim=None)

    @torch.no_grad()
    def _init_memory_bank(self, dim: Union[Sequence[int], None]):
        """Initialize the memory bank if it's empty.

        Args:
            dim:
                The dimension of the which are stored in the bank.

        """
        if dim is None:
            # Feature dimension has been specified when initializing the memory bank.
            size = self.size
        else:
            # Feature dimension was inferred from batch.
            size = (self.size[0], *dim)
        self.bank = torch.randn(size).type_as(self.bank)
        self.bank = torch.nn.functional.normalize(self.bank, dim=-1)
        self.bank_ptr = torch.zeros(1).type_as(self.bank_ptr)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: Tensor):
        """Dequeue the oldest batch and add the latest one

        Args:
            batch:
                The latest batch of keys to add to the memory bank.

        """
        if self.gather_distributed:
            batch = utils.concat_all_gather(batch)

        batch_size = batch.shape[0]
        ptr = int(self.bank_ptr)
        if ptr + batch_size >= self.size[0]:
            self.bank[ptr:] = batch[: self.size[0] - ptr].detach()
            self.bank_ptr[0] = 0
        else:
            self.bank[ptr : ptr + batch_size] = batch.detach()
            self.bank_ptr[0] = ptr + batch_size

    def forward(
        self,
        output: Tensor,
        labels: Union[Tensor, None] = None,
        update: bool = False,
    ):
        """Query memory bank for additional negative samples

        Args:
            output:
                The output of the model.
            labels:
                Should always be None, will be ignored.

        Returns:
            The output if the memory bank is of size 0, otherwise the output
            and the entries from the memory bank.

        """

        # no memory bank, return the output
        if self.size[0] == 0:
            return output, None

        # initialize the memory bank if it is not already done
        if self.bank.ndim == 1:
            dim = output.shape[1:]
            self._init_memory_bank(dim)

        # query and update memory bank
        bank = self.bank.clone().detach()

        # only update memory bank if we later do backward pass (gradient)
        if update:
            self._dequeue_and_enqueue(output)

        return output, bank
