from __future__ import annotations

import os

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from lightly.models.modules.center import Center

_XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if _XFORMERS_ENABLED:
        from xformers.ops import cross_entropy as _xformers_cross_entropy

        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False


class IBOTPatchLoss(Module):
    """Implementation of the iBOT patch loss [0] as used in DINOv2 [1].

    Implementation is based on [2].

    - [0]: iBOT, 2021, https://arxiv.org/abs/2111.07832
    - [1]: DINOv2, 2023, https://arxiv.org/abs/2304.07193
    - [2]: https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/ibot_patch_loss.py

    Attributes:
        output_dim:
            Dimension of the model output.
        teacher_temp:
            Temperature for the teacher output.
        student_temp:
            Temperature for the student output.
        center_mode:
            Mode for center calculation. Only 'mean' is supported.
        center_momentum:
            Momentum term for the center update.
    """

    def __init__(
        self,
        output_dim: int = 65536,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_mode: str = "mean",
        center_momentum: float = 0.9,
    ) -> None:
        """Initializes the iBOTPatchLoss module with the specified parameters."""
        super().__init__()

        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

        self.center = Center(
            size=(1, output_dim),
            mode=center_mode,
            momentum=center_momentum,
        )

    def forward(
        self,
        teacher_out: Tensor,
        student_out: Tensor,
        mask: Tensor,
        teacher_temp: float | None = None,
    ) -> Tensor:
        """Forward pass through the iBOT patch loss.

        Args:
            teacher_out:
                Tensor with shape (batch_size * sequence_length, embed_dim) containing
                the teacher output of the masked tokens.
            student_out:
                Tensor with shape (batch_size * sequence_length, embed_dim) containing
                the student output of the masked tokens.
            mask:
                Boolean tensor with shape (batch_size, height, width) containing the
                token mask. Exactly batch_size * sequence_length entries must be set to
                True in the mask.
            teacher_temp:
                The temperature used for the teacher output. If None, the default
                temperature defined in __init__ is used.

        Returns:
            The loss value.
        """
        # B = batch size, N = sequence length = number of masked tokens, D = embed dim
        # H = height (in tokens), W = width (in tokens)
        # Note that N <= H * W depending on how many tokens are masked.
        teacher_temperature = torch.tensor(
            teacher_temp if teacher_temp is not None else self.teacher_temp
        )

        # Calculate cross-entropy loss.
        teacher_softmax = F.softmax(
            (teacher_out - self.center.value) / teacher_temperature, dim=-1
        )
        student_log_softmax = F.log_softmax(student_out / self.student_temp, dim=-1)

        # (B * N, D) -> (B * N)
        loss = -torch.sum(teacher_softmax * student_log_softmax, dim=-1)

        # Get weights.
        # (B, H, W) -> (B, 1, 1)
        num_masked_per_image = mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        # (B, 1, 1) -> (B, H, W) -> (B * N)
        weight = (1.0 / num_masked_per_image).expand_as(mask)[mask]

        # Apply weighting.
        B = mask.shape[0]
        loss = (loss * weight).sum() / B

        self.center.update(teacher_out)

        return loss


class IBOTPlusPlusPatchLoss(IBOTPatchLoss):
    """Implementation of the iBOT++ patch loss from TIPSv2.

    iBOT++ extends the iBOT masked patch loss by applying patch-level
    self-distillation to all patch tokens, including visible tokens. This is
    useful for dense downstream tasks because all patch features are directly
    anchored to the teacher distribution.

    The loss expects full teacher and student patch logits with shape
    ``(B, N, K)`` where ``B`` is the batch size, ``N`` is the number of patch
    tokens, and ``K`` is the number of prototypes. Inputs with shape
    ``(B * N, K)`` are also supported when ``mask`` is provided to infer the
    batch size.

    Examples:
        >>> criterion = IBOTPlusPlusPatchLoss(output_dim=8192)
        >>> teacher_out = torch.randn(8, 196, 8192)
        >>> student_out = torch.randn(8, 196, 8192)
        >>> loss = criterion(teacher_out=teacher_out, student_out=student_out)
        >>>
        >>> ssl_loss = torch.tensor(1.0)
        >>> loss = ssl_loss + 2.0 * criterion(teacher_out, student_out)

    References:
        TIPSv2, 2026, https://arxiv.org/abs/2604.12012
        iBOT, 2021, https://arxiv.org/abs/2111.07832
    """

    def forward(
        self,
        teacher_out: Tensor,
        student_out: Tensor,
        mask: Tensor | None = None,
        teacher_temp: float | None = None,
    ) -> Tensor:
        """Forward pass through the iBOT++ patch loss.

        Args:
            teacher_out:
                Tensor with shape ``(B, N, K)`` or ``(B * N, K)`` containing
                full patch logits from the teacher model.
            student_out:
                Tensor with the same shape as ``teacher_out`` containing full
                patch logits from the student model.
            mask:
                Optional boolean tensor with shape ``(B, H, W)`` or ``(B, N)``.
                Required when ``teacher_out`` has rank 2 so that the batch size
                ``B`` can be inferred.
            teacher_temp:
                The temperature used for the teacher output. If None, the default
                temperature defined in ``__init__`` is used.

        Returns:
            The loss value as a scalar tensor.

        Raises:
            ValueError:
                If ``teacher_out`` and ``student_out`` shapes differ, if the
                input rank is not 2 or 3, if rank-2 inputs are given without a
                mask, or if the mask shape is incompatible with the input.
        """
        if teacher_out.shape != student_out.shape:
            raise ValueError(
                f"teacher_out and student_out must have the same shape but got "
                f"{tuple(teacher_out.shape)} and {tuple(student_out.shape)}."
            )
        if teacher_out.dim() not in (2, 3):
            raise ValueError(
                f"teacher_out must be rank 2 or 3 but got rank {teacher_out.dim()}."
            )

        if teacher_out.dim() == 3:
            # (B, N, K)
            B = teacher_out.shape[0]
            teacher_flat = teacher_out.flatten(0, 1)
            student_flat = student_out.flatten(0, 1)
        else:
            # (B * N, K) — need mask to recover B
            if mask is None:
                raise ValueError(
                    "mask is required when teacher_out has rank 2 so that the batch "
                    "size B can be inferred."
                )
            B = mask.shape[0]
            BN = teacher_out.shape[0]
            if BN % B != 0:
                raise ValueError(
                    f"teacher_out length {BN} is not divisible by batch size {B} "
                    f"inferred from mask."
                )
            teacher_flat = teacher_out
            student_flat = student_out

        teacher_temperature = torch.tensor(
            teacher_temp if teacher_temp is not None else self.teacher_temp
        )

        # (B * N, K)
        teacher_softmax = F.softmax(
            (teacher_flat - self.center.value) / teacher_temperature, dim=-1
        )

        if XFORMERS_AVAILABLE:
            # Fused kernel avoids materialising the (B*N, K) student log-softmax tensor.
            per_token_loss = _xformers_cross_entropy(
                student_flat.unsqueeze(0).float(),
                teacher_softmax.unsqueeze(0).float(),
                self.student_temp,
                bw_inplace=True,
            ).squeeze(0)
        else:
            per_token_loss = -(
                teacher_softmax * F.log_softmax(student_flat / self.student_temp, dim=-1)
            ).sum(dim=-1)

        loss = per_token_loss.mean()

        self.center.update(teacher_flat)

        return loss
