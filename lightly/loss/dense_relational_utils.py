"""Helpers for the dense relational PatchKernelAlignmentLoss.

These utilities back :class:`~lightly.loss.PatchKernelAlignmentLoss`. The loss
compares the pairwise similarity (kernel) structure of student and teacher dense
features of shape ``(B, N, D)``. The helpers here cover:

- validating dense-feature inputs,
- optionally subsampling tokens to bound the ``O(N^2)`` kernel cost,
- mask-aware double-centering of a kernel matrix.

All math is local to each image, so the helpers are AMP-, CUDA- and DDP-safe
without any cross-rank communication.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.ops import roi_align


def _validate_dense_inputs(
    student_features: Tensor,
    teacher_features: Tensor,
    mask: Tensor | None,
) -> None:
    """Validates the shapes and dtypes of dense relational loss inputs.

    Raises:
        ValueError: If any input has an invalid shape or dtype.
    """
    if student_features.dim() != 3:
        raise ValueError(
            f"student_features must have shape (B, N, D) but has shape "
            f"{tuple(student_features.shape)}."
        )
    if teacher_features.dim() != 3:
        raise ValueError(
            f"teacher_features must have shape (B, N, D) but has shape "
            f"{tuple(teacher_features.shape)}."
        )
    if student_features.shape != teacher_features.shape:
        raise ValueError(
            f"student_features and teacher_features must have the same shape "
            f"but have {tuple(student_features.shape)} and "
            f"{tuple(teacher_features.shape)}."
        )
    if mask is not None:
        batch_size, n_tokens, _ = student_features.shape
        if mask.shape != (batch_size, n_tokens):
            raise ValueError(
                f"mask must have shape {(batch_size, n_tokens)} but has shape "
                f"{tuple(mask.shape)}."
            )
        if mask.dtype != torch.bool:
            raise ValueError(
                f"mask must have dtype torch.bool but has dtype {mask.dtype}."
            )


def _prepare_features(
    student_features: Tensor,
    teacher_features: Tensor,
    mask: Tensor | None,
    max_tokens: int | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Prepares student/teacher features for kernel comparison.

    Detaches the teacher and optionally subsamples a fixed number of (preferably
    valid) tokens per image to bound the ``O(N^2)`` kernel cost.

    Args:
        student_features: Student dense features with shape ``(B, N, D)``.
        teacher_features: Teacher dense features with shape ``(B, N, D)``.
            Detached internally.
        mask: Optional boolean tensor ``(B, N)`` where ``True`` marks tokens to
            ignore (e.g. iBOT-masked positions). ``None`` keeps all tokens.
        max_tokens: If not None and ``N > max_tokens``, randomly subsample
            ``max_tokens`` tokens per image, preferring valid ones.

    Returns:
        A tuple ``(student, teacher, valid)`` where ``student`` and ``teacher``
        have shape ``(B, M, D)`` and ``valid`` is a float tensor ``(B, M)`` with
        ``1.0`` for valid tokens and ``0.0`` for ignored ones (``M = N`` unless
        subsampled).
    """
    student = student_features
    teacher = teacher_features.detach()

    batch_size, n_tokens, _ = student.shape
    if mask is None:
        valid = student.new_ones(batch_size, n_tokens)
    else:
        # mask=True marks tokens to ignore, so valid tokens are ~mask.
        valid = (~mask).to(device=student.device, dtype=student.dtype)

    if max_tokens is not None and n_tokens > max_tokens:
        # Sample max_tokens columns per image, preferring valid rows: invalid
        # tokens get +inf scores so they are picked last.
        scores = torch.rand(batch_size, n_tokens, device=student.device).masked_fill(
            valid == 0, torch.inf
        )
        indices = scores.topk(max_tokens, dim=1, largest=False).indices  # (B, M)
        student = torch.gather(
            student, 1, indices.unsqueeze(-1).expand(-1, -1, student.shape[-1])
        )
        teacher = torch.gather(
            teacher, 1, indices.unsqueeze(-1).expand(-1, -1, teacher.shape[-1])
        )
        valid = torch.gather(valid, 1, indices)

    return student, teacher, valid


def roi_resample_to_grid(
    feature_map: Tensor,
    boxes: Tensor,
    out_h: int,
    out_w: int,
) -> Tensor:
    """Resamples a per-sample ROI of a dense feature map onto a common grid.

    This helper is only needed for the **cross-view** variant of
    :class:`~lightly.loss.PatchKernelAlignmentLoss`. When the student and teacher
    see different (but overlapping) crops of the same image, token ``i`` of one
    view does not describe the same spatial location as token ``i`` of the other,
    so the kernels are not comparable. Resampling each view over its shared region
    onto an identical ``out_h x out_w`` grid restores that correspondence, after
    which the resampled features are passed to the loss as ``(B, N, D)`` with
    ``N = out_h * out_w``. For the **same-view** case (both views already aligned,
    e.g. iBOT-style masking) you call the loss directly and do not need this
    helper.

    Thin, model-agnostic wrapper around :func:`torchvision.ops.roi_align`
    (``aligned=True``, the pixel-center convention for a continuous box mapping);
    differentiable w.r.t. ``feature_map``.

    Example (cross-view):
        >>> import torch
        >>> from lightly.loss import PatchKernelAlignmentLoss, roi_resample_to_grid
        >>> # Patch tokens reshaped back to the (B, C, H, W) patch grid per view.
        >>> student_map = student_tokens.transpose(1, 2).reshape(B, C, H, W)
        >>> teacher_map = teacher_tokens.transpose(1, 2).reshape(B, C, H, W)
        >>> # Shared region of both crops, in feature-grid coordinates (B, 4).
        >>> student = roi_resample_to_grid(student_map, student_boxes, 7, 7)
        >>> teacher = roi_resample_to_grid(teacher_map, teacher_boxes, 7, 7)
        >>> loss = PatchKernelAlignmentLoss()(student, teacher)  # (B, 49, C)

    Args:
        feature_map: Dense features with shape ``(B, C, H, W)`` (e.g. ViT patch
            tokens reshaped back to the patch grid).
        boxes: ROI boxes ``(B, 4)`` as ``(x0, y0, x1, y1)`` in feature-grid
            coordinates (``x`` in ``[0, W]``, ``y`` in ``[0, H]``), one per
            sample, with ``x0 <= x1`` and ``y0 <= y1``.
        out_h: Output grid height.
        out_w: Output grid width.

    Returns:
        Resampled features ``(B, out_h * out_w, C)`` in row-major token order.

    Raises:
        ValueError: If shapes are invalid.
    """
    if feature_map.dim() != 4:
        raise ValueError(
            f"feature_map must have shape (B, C, H, W) but has shape "
            f"{tuple(feature_map.shape)}."
        )
    if boxes.dim() != 2 or boxes.shape[1] != 4:
        raise ValueError(
            f"boxes must have shape (B, 4) but has shape {tuple(boxes.shape)}."
        )
    if boxes.shape[0] != feature_map.shape[0]:
        raise ValueError(
            f"boxes batch size {boxes.shape[0]} does not match feature_map "
            f"batch size {feature_map.shape[0]}."
        )
    batch_index = torch.arange(
        feature_map.shape[0], device=feature_map.device, dtype=feature_map.dtype
    ).unsqueeze(1)
    rois = torch.cat([batch_index, boxes.to(feature_map.dtype)], dim=1)  # (B, 5)
    # roi_align is untyped (returns Any); annotate so the return type checks.
    pooled: Tensor = roi_align(
        feature_map,
        rois,
        output_size=(out_h, out_w),
        spatial_scale=1.0,
        aligned=True,
    )  # (B, C, out_h, out_w)
    return pooled.flatten(2).permute(0, 2, 1).contiguous()  # (B, out_h*out_w, C)


def _linear_kernel(features: Tensor) -> Tensor:
    """Returns the linear kernel ``features @ features^T`` with shape (B, N, N)."""
    return features @ features.transpose(-2, -1)


def _double_center(kernel: Tensor, valid: Tensor) -> Tensor:
    """Mask-aware double-centering of a batch of kernel matrices.

    Computes ``H K H`` with ``H = I - (1/n) 11^T`` restricted to valid tokens,
    i.e. row/column/total means are taken over valid entries only. Entries
    involving an invalid token are zeroed in the output.

    Args:
        kernel: Kernel matrices with shape ``(B, N, N)``.
        valid: Float validity weights with shape ``(B, N)`` (1.0 valid, 0.0
            ignored).

    Returns:
        Centered kernel matrices with shape ``(B, N, N)``.
    """
    w = valid.to(kernel.dtype)
    n = w.sum(dim=-1).clamp(min=1.0)  # (B,)
    # Row means over valid columns (mean over j for each row i).
    row_mean = (kernel * w[:, None, :]).sum(dim=-1) / n[:, None]  # (B, N)
    # Column means over valid rows (mean over i for each column j).
    col_mean = (kernel * w[:, :, None]).sum(dim=1) / n[:, None]  # (B, N)
    total = (kernel * w[:, None, :] * w[:, :, None]).sum(dim=(1, 2)) / (n * n)  # (B,)
    centered = (
        kernel - row_mean[:, :, None] - col_mean[:, None, :] + total[:, None, None]
    )
    return centered * w[:, :, None] * w[:, None, :]


def _reduce_per_image(
    loss_per_image: Tensor,
    valid_image: Tensor,
    reference: Tensor,
) -> Tensor:
    """Mean of the per-image loss over images with enough valid tokens.

    Args:
        loss_per_image: Per-image loss with shape ``(B,)``.
        valid_image: Boolean tensor ``(B,)``; images with too few valid tokens
            are excluded.
        reference: A tensor connected to the student graph, used to build a
            differentiable zero when no image is valid.

    Returns:
        The mean loss. A differentiable zero is returned if no image is valid.
    """
    valid_f = valid_image.to(loss_per_image.dtype)
    loss_per_image = loss_per_image * valid_f
    if not bool(valid_image.any()):
        return (reference * 0.0).sum()
    return loss_per_image.sum() / valid_f.sum().clamp(min=1.0)
