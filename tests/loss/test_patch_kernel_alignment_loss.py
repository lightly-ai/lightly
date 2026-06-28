import pytest
import torch

from lightly.loss.patch_kernel_alignment_loss import PatchKernelAlignmentLoss


class TestPatchKernelAlignmentLoss:
    def test_forward_returns_finite_scalar(self) -> None:
        torch.manual_seed(0)
        criterion = PatchKernelAlignmentLoss()
        loss = criterion(
            student_features=torch.randn(2, 16, 8),
            teacher_features=torch.randn(2, 16, 8),
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_identical_features_give_near_zero_loss(self) -> None:
        # CKA of a feature set with itself is 1, so the loss (1 - CKA) is ~0.
        torch.manual_seed(0)
        criterion = PatchKernelAlignmentLoss()
        features = torch.randn(3, 16, 8)
        loss = criterion(
            student_features=features,
            teacher_features=features.clone(),
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_permuted_teacher_increases_loss(self) -> None:
        # Shuffling the teacher tokens breaks patch-wise alignment, so CKA drops
        # and the loss rises above the aligned baseline.
        torch.manual_seed(0)
        criterion = PatchKernelAlignmentLoss()
        features = torch.randn(2, 16, 8)
        loss_aligned = criterion(
            student_features=features,
            teacher_features=features.clone(),
        )
        permutation = torch.randperm(16)
        loss_permuted = criterion(
            student_features=features,
            teacher_features=features[:, permutation].clone(),
        )
        assert loss_permuted > loss_aligned
