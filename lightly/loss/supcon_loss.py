""" Contrastive Loss Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import distributed as torch_dist
from torch import nn

from lightly.utils import dist


def divide_no_nan(numerator: Tensor, denominator: Tensor) -> Tensor:
    """Performs tensor division, setting result to zero where denominator is zero.

    Args:
        numerator:
            Numerator tensor.
        denominator:
            Denominator tensor with possible zeroes.

    Returns:
        Result with zeros where denominator is zero.
    """
    result = torch.zeros_like(numerator)
    nonzero_mask = denominator != 0
    result[nonzero_mask] = numerator[nonzero_mask] / denominator[nonzero_mask]
    return result


class ContrastMode(Enum):
    """Contrast Mode Enum for SupCon Loss.

    Offers the three contrast modes as enum for the SupCon loss. The three modes are:

    - ContrastMode.ALL: Uses all positives and negatives.
    - ContrastMode.ONE_POSITIVE: Uses only one positive, and all negatives.
    - ContrastMode.ONLY_NEGATIVES: Uses no positives, only negatives.
    """

    ALL = 1
    ONE_POSITIVE = 2
    ONLY_NEGATIVES = 3


VALID_CONTRAST_MODES = set(item.name for item in ContrastMode)


class SupConLoss(nn.Module):
    """Implementation of the Supervised Contrastive Loss.

    This implementation follows the SupCon[0] paper.

    - [0] SupCon, 2020, https://arxiv.org/abs/2004.11362

    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        contrast_mode:
            Whether to use all positives, one positive, or none. All negatives are
            used in all cases.
        gather_distributed:
            If True then negatives from all GPUs are gathered before the
            loss calculation.

    Raises:
        ValueError: If abs(temperature) < 1e-8 to prevent divide by zero.
        ValueError: If gather_distributed is True but torch.distributed is not available.
        ValueError: If contrast_mode is outside the accepted ContrastMode values.

    Examples:
        >>> # initialize loss function
        >>> loss_fn = SupConLoss()
        >>>
        >>> # generate two or more views of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimCLR model
        >>> out0, out1 = model(t0), model(t1)
        >>>
        >>> # Stack views along 2nd dimensions
        >>> features = torch.stack([out0, out1], dim=1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(features, labels)

    """

    def __init__(
        self,
        temperature: float = 0.5,
        contrast_mode: ContrastMode = ContrastMode.ALL,
        gather_distributed: bool = False,
    ):
        """Initializes the SupConLoss module with the specified parameters.

        Args:
            temperature:
                 Scale logits by the inverse of the temperature.
            contrast_mode:
                Whether to use all positives, one positive, or none. All negatives are
                used in all cases.
            gather_distributed:
                 If True, negatives from all GPUs are gathered before the loss calculation.

        Raises:
            ValueError: If temperature is less than 1e-8 to prevent divide by zero.
            ValueError: If gather_distributed is True but torch.distributed is not available.
            ValueError: If contrast_mode is outside the accepted ContrastMode values.
        """
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.positives_cap = -1  # Unused at the moment
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

        if contrast_mode.name not in VALID_CONTRAST_MODES:
            raise ValueError(
                f"contrast_mode is {contrast_mode} but must be one of ContrastMode.{VALID_CONTRAST_MODES}"
            )

    def forward(self, features: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        """Forward pass through Supervised Contrastive Loss.

        Computes the loss based on contrast_mode setting.

        Args:
            features:
                Tensor of at least 3 dimensions, corresponding to
                (batch_size, num_views, ...)
            labels:
                Onehot labels for each sample. Must match shape
                (batch_size, num_classes)

        Raises:
            ValueError: If features does not have at least 3 dimensions.
            ValueError: If number of labels does not match batch_size.
            ValueError: If labels is not one-hot encoded.

        Returns:
            Supervised Contrastive Loss value.
        """

        if len(features.shape) < 3:
            raise ValueError(
                f"Features must have at least 3 dimensions, got {len(features.shape)}."
            )

        device = features.device
        batch_size, num_views = features.shape[:2]

        if labels is not None and labels.size(0) != batch_size:
            raise ValueError(
                f"When setting labels, labels must match batch_size {batch_size}, got {labels.size(0)}."
            )

        if labels is not None:
            if not self._is_one_hot(labels):
                raise ValueError(
                    "labels must be a 2D matrix representing the one-hot encoded classes."
                )

        # Flatten the features in case they are still images or other
        features = features.flatten(2)

        # Normalize the features to length 1
        features = F.normalize(features, dim=2)

        # Memory bank could be used here but labelled samples are not yet supported.

        # Use cosine similarity (dot product) as all vectors are normalized to unit length

        # Use other samples from different classes in batch as negatives
        # and create diagonal mask that only selects similarities between
        # views of the same image / same class
        if self.gather_distributed and dist.world_size() > 1:
            # Gather hidden representations and optional labels from other processes
            global_features = torch.cat(dist.gather(features), 0)
            diag_mask = dist.eye_rank(batch_size, device=device)
            if labels is not None:
                global_labels = torch.cat(dist.gather(labels), 0)
        else:
            # Single process
            global_features = features
            diag_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
            if labels is not None:
                global_labels = labels

        # Use the diagonal mask if labels is none, else compute the mask based on labels
        if labels is None:
            # No labels, typical semi-supervised contrastive learning like SimCLR
            mask = diag_mask
        else:
            mask = (labels @ global_labels.T).to(device)

        # Get features in shape [num_views * batch_size, c]
        all_global_features = global_features.permute(1, 0, 2).reshape(
            -1, global_features.size(-1)
        )

        if self.contrast_mode == ContrastMode.ONE_POSITIVE:
            # We take only the first view as anchor
            anchor_features = features[:, 0]
            num_anchor_views = 1
        else:
            # We take all views as anchors in the same shape as the global features
            anchor_features = features.permute(1, 0, 2).reshape(-1, features.size(-1))
            num_anchor_views = num_views

        # Obtain the logits between anchor features and features across all processes
        # Logits will be shaped [local_batch_size * num_anchor_views, global_batch_size * num_views]
        # We then temperature scale it and subtract the max to improve numerical stability
        # In the einsum, n is local_batch_size * num_anchor_views, m is global_batch_size * num_views,
        # and c is the flattened feature length
        # Note: features are ordered by view first, i.e. first all samples of view 0, then all samples
        # of view 1, and so on.
        logits = torch.einsum("nc,mc->nm", anchor_features, all_global_features)
        logits /= self.temperature
        logits -= logits.max(dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(logits)

        # Get the positive and negative masks for numerator & denominator
        positives_mask, negatives_mask = self._create_tiled_masks(
            mask.long(),
            diag_mask.long(),
            num_views,
            num_anchor_views,
            self.positives_cap,
        )
        num_positives_per_row = positives_mask.sum(dim=1)

        # Calculate denominator based on contrast_mode
        if self.contrast_mode == ContrastMode.ONE_POSITIVE:
            denominator = exp_logits + (exp_logits * negatives_mask).sum(
                dim=1, keepdim=True
            )
        elif self.contrast_mode == ContrastMode.ALL:
            denominator = (exp_logits * negatives_mask).sum(dim=1, keepdim=True)
            denominator += (exp_logits * positives_mask).sum(dim=1, keepdim=True)
        else:  # ContrastMode.ONLY_NEGATIVES
            denominator = (exp_logits * negatives_mask).sum(dim=1, keepdim=True)

        # num_positives_per_row can be zero iff 1 view is used. Here we use a safe
        # dividing method seting those values to zero to prevent division by zero errors.

        # Only implements SupCon_{out}.
        log_probs = (logits - torch.log(denominator)) * positives_mask
        log_probs = log_probs.sum(dim=1)
        log_probs = divide_no_nan(log_probs, num_positives_per_row)

        loss = -log_probs

        # Adjust for num_positives_per_row being zero when using exactly 1 view
        if num_views != 1:
            loss = loss.mean(dim=0)
        else:
            num_valid_views_per_sample = num_positives_per_row.unsqueeze(0)
            loss = divide_no_nan(loss, num_valid_views_per_sample).squeeze()

        return loss

    def _create_tiled_masks(
        self,
        untiled_mask: Tensor,
        diagonal_mask: Tensor,
        num_views: int,
        num_anchor_views: int,
        positives_cap: int,
    ) -> Tuple[Tensor, Tensor]:
        # Get total batch size across all processes
        global_batch_size = untiled_mask.size(1)

        # Find index of the anchor for each sample
        labels = torch.argmax(diagonal_mask, dim=1)

        # Generate tiled labels across views
        tiled_labels = []
        for i in range(num_anchor_views):
            tiled_labels.append(labels + global_batch_size * i)
        tiled_labels_tensor = torch.cat(tiled_labels, 0)
        tiled_diagonal_mask = F.one_hot(
            tiled_labels_tensor, global_batch_size * num_views
        )

        # Mask to zero the diagonal at the end
        all_but_diagonal_mask = 1 - tiled_diagonal_mask

        # All tiled positives
        uncapped_positives_mask = torch.tile(
            untiled_mask, [num_anchor_views, num_views]
        )

        # The negatives is simply the bitflipped positives
        negatives_mask = 1.0 - uncapped_positives_mask

        # For when positives_cap is implemented
        if positives_cap > -1:
            raise NotImplementedError("Capping positives is not yet implemented.")
        else:
            positives_mask = uncapped_positives_mask

        # Zero out the self-contrast
        positives_mask *= all_but_diagonal_mask

        return positives_mask, negatives_mask

    def _is_one_hot(self, tensor: Tensor) -> bool:
        # Tensor is not a 2D matrix
        if tensor.ndim != 2:
            return False

        # Check values are only 0 or 1
        is_binary = ((tensor == 0) | (tensor == 1)).all()

        # Check each row sums to 1
        row_sums = tensor.sum(dim=1)
        has_single_one = (row_sums == 1).all()

        return bool(is_binary.item() and has_single_one.item())
