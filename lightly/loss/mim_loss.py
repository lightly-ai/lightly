# Copyright (c) 2024. Lightly AG and its affiliates.
# All Rights Reserved

from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MaskedImageModelingLoss(nn.Module):
    """Loss for Masked Image Modeling as described in BEIT.

    Computes the cross-entropy between the predicted visual token logits at
    masked patch positions and the target discrete visual token indices
    produced by the image tokenizer (discrete VAE).

    The objective is:

        L = -E[ sum_{i in M} log p_MIM(z_i | x_M) ]

    where M is the set of masked positions, z_i is the target visual token,
    and x_M is the corrupted (masked) image.


    Attributes:
        reduction:
            Specifies how to aggregate the per-token losses across the batch.
            Must be one of ``"mean"`` (default) or ``"sum"``.
            ``"mean"`` divides by the total number of masked tokens across
            the batch; ``"sum"`` returns the raw sum.
        label_smoothing:
            Amount of label smoothing in [0.0, 1.0) applied to the
            cross-entropy. Matches the ``label_smoothing`` argument of
            :func:`torch.nn.functional.cross_entropy`. Default: 0.0.

    Examples:
        >>> import torch
        >>> from lightly.loss import MaskedImageModelingLoss
        >>>
        >>> loss_fn = MaskedImageModelingLoss()
        >>>
        >>> # mim_logits: predictions at masked positions  (N_masked, vocab_size)
        >>> # token_targets: ground-truth token ids        (N_masked,)
        >>> mim_logits = torch.randn(40, 8192)
        >>> token_targets = torch.randint(0, 8192, (40,))
        >>> loss = loss_fn(mim_logits, token_targets)
    """

    def __init__(
        self,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"Invalid reduction '{reduction}'. Must be 'mean' or 'sum'."
            )
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}."
            )

        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        mim_logits: Tensor,
        token_targets: Tensor,
    ) -> Tensor:
        """Compute the MIM loss.

        Args:
            mim_logits:
                Predicted logits for each masked patch position.
                Shape: ``(N_masked, vocab_size)`` where ``N_masked`` is the
                total number of masked patches across the batch (i.e.
                ``batch_size * num_masked_patches_per_image`` when the mask
                count is fixed, or the flattened variable-length masked set).
            token_targets:
                Ground-truth visual token indices from the image tokenizer for
                each masked position. Shape: ``(N_masked,)``, dtype ``long``.

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError:
                If ``mim_logits`` and ``token_targets`` have incompatible
                leading dimensions.
        """
        if mim_logits.shape[0] != token_targets.shape[0]:
            raise ValueError(
                f"mim_logits has {mim_logits.shape[0]} entries but "
                f"token_targets has {token_targets.shape[0]}. "
                "Both must have the same number of masked positions."
            )

        return F.cross_entropy(
            mim_logits,
            token_targets,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
