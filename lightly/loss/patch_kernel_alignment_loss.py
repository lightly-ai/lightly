from __future__ import annotations

from torch import Tensor
from torch.nn import Module

from lightly.loss.dense_relational_utils import (
    _double_center,
    _linear_kernel,
    _prepare_features,
    _reduce_per_image,
    _validate_dense_inputs,
)

# Clamp before sqrt in the CKA denominator so the gradient stays finite when an
# image's kernel is (near-)constant (HSIC -> 0); for non-degenerate images this
# is a no-op.
_EPS = 1e-6


class PatchKernelAlignmentLoss(Module):
    """Patch kernel alignment loss for dense self-supervised learning.

    Aligns the *relational* structure of student and teacher dense features
    rather than asking for exact patch correspondence. For each image it builds
    the pairwise linear kernels of the student and teacher tokens and maximizes
    their Centered Kernel Alignment (CKA):

        K_s = Z_s Z_s^T,  K_t = Z_t Z_t^T
        CKA = <H K_s H, H K_t H>_F / (||H K_s H||_F * ||H K_t H||_F)
        L = 1 - CKA

    where ``H = I - (1/N) 11^T`` is the centering matrix. CKA is invariant to
    isotropic scaling and orthogonal transforms of the features, so the loss
    captures region grouping and object-level organization without requiring
    nearest-neighbour sorting, clustering, prototypes, or geometric patch
    identity. This makes it well suited to detection and segmentation.

    The loss is model-agnostic: it operates on dense features ``(B, N, D)``
    where ``N`` is the number of ViT patch tokens (ConvNet features must be
    flattened to ``(B, N, D)`` by the caller). Teacher features are detached
    internally. Geometry/ROI alignment for a cross-view variant is the caller's
    responsibility; the loss receives already aligned dense features.

    Args:
        max_tokens:
            If not None and ``N > max_tokens``, randomly subsample
            ``max_tokens`` tokens per image (preferring valid ones) before
            forming the kernel, to bound the ``O(B N^2 D)`` cost.

    Example:
        >>> criterion = PatchKernelAlignmentLoss(max_tokens=512)
        >>> loss = criterion(
        ...     student_features=student_features,  # (B, N, D)
        ...     teacher_features=teacher_features,  # (B, N, D)
        ... )
    """

    def __init__(self, max_tokens: int | None = None) -> None:
        """Initializes the PatchKernelAlignmentLoss module.

        Raises:
            ValueError: If ``max_tokens`` is not ``None`` and ``< 2``.
        """
        super().__init__()
        if max_tokens is not None and max_tokens < 2:
            raise ValueError(f"max_tokens must be >= 2 or None but is {max_tokens}.")
        self.max_tokens = max_tokens

    def forward(
        self,
        student_features: Tensor,
        teacher_features: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Computes the patch kernel alignment loss.

        Args:
            student_features:
                Student dense features with shape ``(B, N, D)``.
            teacher_features:
                Teacher dense features with shape ``(B, N, D)``. Detached
                internally.
            mask:
                Optional boolean tensor ``(B, N)`` where ``True`` marks tokens
                to ignore. Images with fewer than two valid tokens do not
                contribute to the loss.

        Returns:
            The mean loss over images with enough valid tokens, as a scalar
            tensor. A differentiable zero is returned if no image qualifies.

        Raises:
            ValueError: If any input has an invalid shape or dtype.
        """
        _validate_dense_inputs(student_features, teacher_features, mask)

        student, teacher, valid = _prepare_features(
            student_features=student_features,
            teacher_features=teacher_features,
            mask=mask,
            max_tokens=self.max_tokens,
        )

        # Compute the kernels and CKA in float32. Under bf16/fp16 the HSIC terms
        # (sums of products of small centered-kernel entries) can underflow to 0,
        # and the backward of sqrt(0) is +inf -> NaN gradients. This bites the
        # cross-view variant where ROI-aligned regions can be small/near-constant.
        student = student.float()
        teacher = teacher.float()
        valid_f = valid.float()
        k_s = _double_center(_linear_kernel(student), valid_f)
        k_t = _double_center(_linear_kernel(teacher), valid_f)

        hsic_st = (k_s * k_t).sum(dim=(1, 2))
        hsic_ss = (k_s * k_s).sum(dim=(1, 2))
        hsic_tt = (k_t * k_t).sum(dim=(1, 2))
        denom = hsic_ss.clamp_min(_EPS).sqrt() * hsic_tt.clamp_min(_EPS).sqrt()
        cka = hsic_st / denom
        loss_per_image = 1.0 - cka

        valid_image = valid.sum(dim=-1) >= 2
        return _reduce_per_image(
            loss_per_image=loss_per_image,
            valid_image=valid_image,
            reference=student_features,
        )
