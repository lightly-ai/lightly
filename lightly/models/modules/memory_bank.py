""" Memory Bank Wrapper """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch

from lightly.models import utils


class MemoryBankModule(torch.nn.Module):
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
        >>>     def forward(self, output: torch.Tensor,
        >>>                 labels: torch.Tensor = None):
        >>>
        >>>         output, negatives = super(
        >>>             MyLossFunction, self).forward(output)
        >>>
        >>>         if negatives is not None:
        >>>             # evaluate loss with negative samples
        >>>         else:
        >>>             # evaluate loss without negative samples

    """

    def __init__(self, size: int = 65536, gather_distributed: bool = False):
        super(MemoryBankModule, self).__init__()

        if size < 0:
            msg = f"Illegal memory bank size {size}, must be non-negative."
            raise ValueError(msg)

        self.size = size
        self.gather_distributed = gather_distributed
        self.register_buffer(
            "bank", tensor=torch.empty(0, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "bank_ptr", tensor=torch.empty(0, dtype=torch.long), persistent=False
        )

    @torch.no_grad()
    def _init_memory_bank(self, dim: int):
        """Initialize the memory bank if it's empty

        Args:
            dim:
                The dimension of the which are stored in the bank.

        """
        # create memory bank
        # we could use register buffers like in the moco repo
        # https://github.com/facebookresearch/moco but we don't
        # want to pollute our checkpoints
        self.bank = torch.randn(dim, self.size).type_as(self.bank)
        self.bank = torch.nn.functional.normalize(self.bank, dim=0)
        self.bank_ptr = torch.zeros(1).type_as(self.bank_ptr)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: torch.Tensor):
        """Dequeue the oldest batch and add the latest one

        Args:
            batch:
                The latest batch of keys to add to the memory bank.

        """
        if self.gather_distributed:
            batch = utils.concat_all_gather(batch)
        batch_size = batch.shape[0]
        ptr = int(self.bank_ptr)

        if ptr + batch_size >= self.size:
            self.bank[:, ptr:] = batch[: self.size - ptr].T.detach()
            self.bank_ptr[0] = 0
        else:
            self.bank[:, ptr : ptr + batch_size] = batch.T.detach()
            self.bank_ptr[0] = ptr + batch_size

    def forward(
        self, output: torch.Tensor, labels: torch.Tensor = None, update: bool = False
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
        if self.size == 0:
            return output, None

        _, dim = output.shape

        # initialize the memory bank if it is not already done
        if self.bank.nelement() == 0:
            self._init_memory_bank(dim)

        # query and update memory bank
        bank = self.bank.clone().detach()

        # only update memory bank if we later do backward pass (gradient)
        if update:
            self._dequeue_and_enqueue(output)

        return output, bank
