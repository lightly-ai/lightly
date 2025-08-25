""" Contrastive Loss Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import distributed as torch_dist
from torch import nn

from lightly.utils import dist


class SupConLoss(nn.Module):
    """Implementation of the Supervised Contrastive Loss.

    This implementation follows the SupCon[0] paper.

    - [0] SupCon, 2020, https://arxiv.org/abs/2004.11362

    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        gather_distributed:
            If True then negatives from all GPUs are gathered before the
            loss calculation. If a memory bank is used and gather_distributed is True,
            then tensors from all gpus are gathered before the memory bank is updated.
        rescale:
            Optionally rescale final loss by the temperature for stability.
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

    def __init__(
        self,
        temperature: float = 0.5,
        gather_distributed: bool = False,
        rescale: bool = True,
    ):
        """Initializes the SupConLoss module with the specified parameters.

        Args:
            temperature:
                 Scale logits by the inverse of the temperature.
            gather_distributed:
                 If True, negatives from all GPUs are gathered before the loss calculation.
            rescale:
                Optionally rescale final loss by the temperature for stability.

        Raises:
            ValueError: If temperature is less than 1e-8 to prevent divide by zero.
            ValueError: If gather_distributed is True but torch.distributed is not available.
        """
        super().__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.rescale = rescale
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

    def forward(
        self, out0: Tensor, out1: Tensor, labels: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass through Supervised Contrastive Loss.

        Computes the loss based on contrast_mode setting.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)
            labels:
                Onehot labels for each sample. Must be a vector of length `batch_size`.

        Returns:
            Supervised Contrastive Loss value.
        """
        # Stack the views for efficient computation
        # Allows for more views to be added easily
        features = (out0, out1)
        n_views = len(features)
        out_small = torch.vstack(features)

        device = out_small.device
        batch_size = out_small.shape[0] // n_views

        # Normalize the output to length 1
        out_small = nn.functional.normalize(out_small, dim=1)

        # Gather hidden representations from other processes if distributed
        # and compute the diagonal self-contrast mask
        if self.gather_distributed and dist.world_size() > 1:
            out_large = torch.cat(dist.gather(out_small), 0)
            diag_mask = dist.eye_rank(n_views * batch_size, device=device)
        else:
            # Single process
            out_large = out_small
            diag_mask = torch.eye(n_views * batch_size, device=device, dtype=torch.bool)

        # Use cosine similarity (dot product) as all vectors are normalized to unit length
        # Calculate similiarities
        logits = out_small @ out_large.T
        logits /= self.temperature

        # Set self-similarities to infinitely small value
        logits[diag_mask] = -1e9

        # Create labels if None
        if labels is None:
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            if self.gather_distributed:
                labels = labels + dist.rank() * batch_size
        labels = labels.repeat(n_views)

        # Soft labels are 0 unless the logit represents a similarity
        # between two of the same classes. We manually set self-similarity
        # (same view of the same item) to 0. When not 0, the value is
        # 1 / n, where n is the number of positive samples
        # (different views of the same item, and all views of other items sharing
        # classes with the item)
        soft_labels = torch.eq(labels, labels.view(-1, 1)).float()
        soft_labels.fill_diagonal_(0.0)
        soft_labels /= soft_labels.sum(dim=1)

        # Compute log probabilities
        log_proba = F.log_softmax(logits, dim=-1)

        # Compute soft cross-entropy loss
        loss = (soft_labels * log_proba).sum(-1)
        loss = -loss.mean()

        # Optional: rescale for stable training
        if self.rescale:
            loss *= self.temperature

        return loss
