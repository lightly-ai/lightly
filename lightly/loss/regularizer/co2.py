""" CO2 Regularizer """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import Sequence, Union

import torch
from torch import Tensor
from torch.nn import Module

from lightly.models.modules.memory_bank import MemoryBankModule


class CO2Regularizer(Module):
    """Implementation of the CO2 regularizer [0] for self-supervised learning.

    - [0] CO2, 2021, https://arxiv.org/abs/2010.02217

    Attributes:
        alpha:
            Weight of the regularization term.
        t_consistency:
            Temperature used during softmax calculations.
        memory_bank_size:
            Size of the memory bank as (num_features, dim) tuple. num_features is the
            number of negatives stored in the bank. If set to 0, the memory bank is
            disabled. Deprecated: If only a single integer is passed, it is interpreted
            as the number of features and the feature dimension is inferred from the
            first batch stored in the memory bank. Leaving out the feature dimension
            might lead to errors in distributed training.

    Examples:
        >>> # initialize loss function for MoCo
        >>> loss_fn = NTXentLoss(memory_bank_size=(4096, 128))
        >>>
        >>> # initialize CO2 regularizer
        >>> co2 = CO2Regularizer(alpha=1.0, memory_bank_size=(4096, 128))
        >>>
        >>> # generate two random trasnforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through the MoCo model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss and apply regularizer
        >>> loss = loss_fn(out0, out1) + co2(out0, out1)
    """

    def __init__(
        self,
        alpha: float = 1,
        t_consistency: float = 0.05,
        memory_bank_size: Union[int, Sequence[int]] = 0,
    ):
        """Initializes the CO2Regularizer with the specified parameters.

        Args:
            alpha:
                Weight of the regularization term.
            t_consistency:
                Temperature used during softmax calculations.
            memory_bank_size:
                Size of the memory bank.
        """
        super().__init__()
        self.memory_bank = MemoryBankModule(size=memory_bank_size)
        # Try-catch the KLDivLoss construction for backwards compatability
        self.log_target = True
        try:
            self.kl_div = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        except TypeError:
            self.log_target = False
            self.kl_div = torch.nn.KLDivLoss(reduction="batchmean")

        self.t_consistency = t_consistency
        self.alpha = alpha

    def forward(self, out0: Tensor, out1: Tensor) -> Tensor:
        """Computes the CO2 regularization term for two model outputs.

        Args:
            out0:
                Output projections of the first set of transformed images.
            out1:
                Output projections of the second set of transformed images.

        Returns:
            The regularization term multiplied by the weight factor alpha.
        """

        # Normalize the output to length 1
        out0 = torch.nn.functional.normalize(out0, dim=1)
        out1 = torch.nn.functional.normalize(out1, dim=1)

        # Update the memory bank with out1 and get negatives(if memory bank size > 0)
        # If the memory_bank size is 0, negatives will be None
        out1, negatives = self.memory_bank.forward(out1, update=True)

        # Get log probabilities
        p = self._get_pseudo_labels(out0, out1, negatives)
        q = self._get_pseudo_labels(out1, out0, negatives)

        # Calculate symmetrized Kullback-Leibler divergence
        if self.log_target:
            div = self.kl_div(p, q) + self.kl_div(q, p)
        else:
            # Can't use log_target because of early torch version
            div = self.kl_div(p, torch.exp(q)) + self.kl_div(q, torch.exp(p))

        return torch.tensor(self.alpha * 0.5 * div)

    def _get_pseudo_labels(
        self, out0: Tensor, out1: Tensor, negatives: Union[Tensor, None] = None
    ) -> Tensor:
        """Computes the soft pseudo labels across negative samples.

        Args:
            out0:
                Output projections of the first set of transformed images (query).
                Shape: bsz x n_ftrs
            out1:
                Output projections of the second set of transformed images (positive sample).
                Shape: bsz x n_ftrs
            negatives:
                Negative samples to compare against. If this is None, the second
                batch of images will be used as negative samples.
                Shape: memory_bank_size x n_ftrs

        Returns:
            Log probability that a positive samples will classify each negative
            sample as the positive sample.
            Shape: bsz x (bsz - 1) or bsz x memory_bank_size
        """
        batch_size, _ = out0.shape
        if negatives is None:
            # Use second batch as negative samples
            # l_pos has shape bsz x 1 and l_neg has shape bsz x bsz
            l_pos = torch.einsum("nc,nc->n", [out0, out1]).unsqueeze(-1)
            l_neg = torch.einsum("nc,ck->nk", [out0, out1.t()])

            # Remove elements on the diagonal
            # l_neg has shape bsz x (bsz - 1)
            l_neg = l_neg.masked_select(
                ~torch.eye(batch_size, dtype=torch.bool, device=l_neg.device)
            ).view(batch_size, batch_size - 1)
        else:
            # Use memory bank as negative samples
            # l_pos has shape bsz x 1 and l_neg has shape bsz x memory_bank_size
            negatives = negatives.to(out0.device)
            l_pos = torch.einsum("nc,nc->n", [out0, out1]).unsqueeze(-1)
            l_neg = torch.einsum("nc,ck->nk", [out0, negatives.clone().detach()])

        # Concatenate such that positive samples are at index 0
        logits = torch.cat([l_pos, l_neg], dim=1)
        # Divide by temperature
        logits = logits / self.t_consistency

        # The input to kl_div is expected to be log(p)
        return torch.nn.functional.log_softmax(logits, dim=-1)
