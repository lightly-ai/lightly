from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from lightly.models.modules import center as center_module
from lightly.models.modules.center import Center

if TYPE_CHECKING:
    from typing import Literal


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
            Mode used to normalize the teacher output. Either 'mean' for the mean
            centering from DINO or 'sinkhorn_knopp' for the Sinkhorn-Knopp centering
            that is optional in DINOv2 and used by DINOv3.
        center_momentum:
            Momentum term for the center update.
        sinkhorn_iterations:
            Number of Sinkhorn-Knopp iterations. Only used if
            `center_mode="sinkhorn_knopp"`.
    """

    def __init__(
        self,
        output_dim: int = 65536,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_mode: Literal["mean", "sinkhorn_knopp"] = "mean",
        center_momentum: float = 0.9,
        sinkhorn_iterations: int = 3,
    ) -> None:
        """Initializes the iBOTPatchLoss module with the specified parameters.

        Raises:
            ValueError: If an unknown center mode is provided.
            ValueError: If sinkhorn_iterations is less than 1.
        """
        super().__init__()

        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

        if center_mode == "mean":
            tracks_center = True
        elif center_mode == "sinkhorn_knopp":
            # Sinkhorn-Knopp centering does not track a center.
            tracks_center = False
        else:
            raise ValueError(
                f"Unknown mode '{center_mode}'. Valid modes are 'mean' and "
                "'sinkhorn_knopp'."
            )
        if sinkhorn_iterations < 1:
            raise ValueError(
                f"sinkhorn_iterations must be at least 1 but is {sinkhorn_iterations}."
            )
        self.center_mode = center_mode
        self.sinkhorn_iterations = sinkhorn_iterations

        # The Center module is still created, with the default mode, to keep the state
        # dict independent of the center mode.
        self.center = Center(
            size=(1, output_dim),
            mode=center_mode if tracks_center else "mean",
            momentum=center_momentum,
        )

    def _teacher_probabilities(
        self, teacher_out: Tensor, teacher_temp: Tensor
    ) -> Tensor:
        """Returns the sharpened teacher probabilities for the given teacher output.

        Applies either mean centering followed by a softmax, or Sinkhorn-Knopp
        centering, depending on center_mode.

        Args:
            teacher_out:
                Tensor with shape (num_tokens, output_dim) containing the teacher
                output.
            teacher_temp:
                The temperature used for the teacher output.

        Returns:
            Tensor with the same shape as teacher_out containing probabilities that
            sum to one along the last dimension. Sinkhorn-Knopp probabilities are
            detached from the computation graph, following the reference
            implementation.
        """
        if self.center_mode == "sinkhorn_knopp":
            # Sinkhorn-Knopp calculates in float32, the probabilities are cast back to
            # keep the dtype of the loss independent of the center mode.
            probabilities = center_module.sinkhorn_knopp(
                x=teacher_out,
                temperature=teacher_temp,
                num_iterations=self.sinkhorn_iterations,
            )
            return probabilities.to(teacher_out.dtype)
        return F.softmax((teacher_out - self.center.value) / teacher_temp, dim=-1)

    def _update_center(self, teacher_out: Tensor) -> None:
        """Updates the center with the given teacher output.

        Does nothing if no center is tracked, which is the case for Sinkhorn-Knopp
        centering.

        Args:
            teacher_out:
                Tensor with shape (num_tokens, output_dim) containing the teacher
                output.
        """
        if self.center_mode == "sinkhorn_knopp":
            return
        self.center.update(teacher_out)

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
        teacher_softmax = self._teacher_probabilities(
            teacher_out=teacher_out, teacher_temp=teacher_temperature
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

        self._update_center(teacher_out)

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
        visible_loss_weight: float = 1.0,
    ) -> Tensor:
        """Forward pass through the iBOT++ patch loss.

        When ``mask`` is provided, the per-token cross-entropy is split into a
        masked term and a visible term. The masked term is normalized by the
        number of masked tokens per image (matching the original iBOT masked
        image modeling signal), and the visible term is normalized by the number
        of visible tokens per image and scaled by ``visible_loss_weight``::

            loss = mean_b(masked_b + visible_loss_weight * visible_b)

        Setting ``visible_loss_weight=0`` recovers the exact iBOT behavior, while
        ``visible_loss_weight>0`` adds the iBOT++ supervision on visible tokens
        without diluting the masked-token signal. When ``mask`` is ``None`` the
        loss falls back to a plain mean over all patch tokens.

        Args:
            teacher_out:
                Tensor with shape ``(B, N, K)`` or ``(B * N, K)`` containing
                full patch logits from the teacher model.
            student_out:
                Tensor with the same shape as ``teacher_out`` containing full
                patch logits from the student model.
            mask:
                Optional boolean tensor with shape ``(B, H, W)`` or ``(B, N)``
                where ``True`` marks masked tokens. Used to weight masked vs.
                visible tokens, and required when ``teacher_out`` has rank 2 so
                that the batch size ``B`` can be inferred.
            teacher_temp:
                The temperature used for the teacher output. If None, the default
                temperature defined in ``__init__`` is used.
            visible_loss_weight:
                Weight applied to the visible-token (unmasked) loss term. Only
                used when ``mask`` is provided. Defaults to ``1.0``. Use ``0.0``
                to recover the original iBOT masked-only behavior.

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

        N = teacher_flat.shape[0] // B

        teacher_temperature = torch.tensor(
            teacher_temp if teacher_temp is not None else self.teacher_temp
        )

        # (B * N, K)
        teacher_softmax = self._teacher_probabilities(
            teacher_out=teacher_flat, teacher_temp=teacher_temperature
        )
        student_log_softmax = F.log_softmax(student_flat / self.student_temp, dim=-1)

        # Per-token cross-entropy: (B * N,) -> (B, N)
        ce = -torch.sum(teacher_softmax * student_log_softmax, dim=-1).view(B, N)

        if mask is None:
            # No mask: average equally over all patch tokens.
            loss = ce.mean()
        else:
            # Split into masked and visible terms. The masked term keeps the
            # original iBOT normalization (per-image mean over masked tokens) so
            # the masked-token signal is not diluted by the visible tokens.
            mask_flat = mask.reshape(B, N).to(dtype=ce.dtype)  # 1.0 = masked
            n_masked = mask_flat.sum(dim=1).clamp(min=1.0)
            n_visible = (1.0 - mask_flat).sum(dim=1).clamp(min=1.0)
            masked_loss = (ce * mask_flat).sum(dim=1) / n_masked
            visible_loss = (ce * (1.0 - mask_flat)).sum(dim=1) / n_visible
            loss = (masked_loss + visible_loss_weight * visible_loss).mean()

        self._update_center(teacher_flat)

        return loss
