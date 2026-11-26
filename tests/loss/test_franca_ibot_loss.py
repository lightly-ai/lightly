from __future__ import annotations

from typing import Callable

import pytest
import torch

from lightly.loss.franca_loss import FrancaIBOTPatchLoss
from lightly.loss.ibot_loss import IBOTPatchLoss


def _ibot_inputs(
    batch_size: int, height: int, width: int, dims: list[int], num_masked_per_image: int
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Builds masked teacher/student patch tokens per level and a matching mask."""
    mask = torch.zeros(batch_size, height * width, dtype=torch.bool)
    mask[:, :num_masked_per_image] = True
    mask = mask.reshape(batch_size, height, width)
    total = int(mask.sum())
    teacher = [torch.randn(total, dim) for dim in dims]
    student = [torch.randn(total, dim) for dim in dims]
    return teacher, student, mask


class TestFrancaIBOTPatchLoss:
    def test_forward_and_backward__mean(self) -> None:
        torch.manual_seed(0)
        dims = [8, 16, 24]
        loss_fn = FrancaIBOTPatchLoss(output_dims=dims, center_mode="mean")
        teacher, student, mask = _ibot_inputs(2, 4, 4, dims, num_masked_per_image=3)
        student = [s.requires_grad_(True) for s in student]
        loss = loss_fn(teacher, student, mask)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        for tensor in student:
            assert tensor.grad is not None

    def test_forward_and_backward__sinkhorn(self) -> None:
        torch.manual_seed(0)
        dims = [8, 16, 24]
        loss_fn = FrancaIBOTPatchLoss(output_dims=dims, center_mode="sinkhorn")
        teacher, student, mask = _ibot_inputs(2, 4, 4, dims, num_masked_per_image=3)
        student = [s.requires_grad_(True) for s in student]
        loss = loss_fn(teacher, student, mask)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        for tensor in student:
            assert tensor.grad is not None

    def test_single_level_matches_ibot_patch_loss(self) -> None:
        """A single-level FrancaIBOTPatchLoss (mean) reduces exactly to IBOTPatchLoss."""
        torch.manual_seed(0)
        dim = 16
        teacher, student, mask = _ibot_inputs(2, 4, 4, [dim], num_masked_per_image=3)

        ibot = IBOTPatchLoss(output_dim=dim)
        franca = FrancaIBOTPatchLoss(output_dims=[dim], center_mode="mean")

        loss_ibot = ibot(teacher[0], student[0], mask)
        loss_franca = franca(teacher, student, mask)

        assert torch.allclose(loss_ibot, loss_franca)
        assert torch.allclose(ibot.center.value, franca.centers[0].get_buffer("center"))

    def test_levels_are_summed_unweighted(self) -> None:
        torch.manual_seed(0)
        dim = 12
        teacher, student, mask = _ibot_inputs(2, 4, 4, [dim], num_masked_per_image=3)

        one = FrancaIBOTPatchLoss(output_dims=[dim], center_mode="mean")
        three = FrancaIBOTPatchLoss(output_dims=[dim, dim, dim], center_mode="mean")

        loss_one = one(teacher, student, mask)
        loss_three = three(teacher * 3, student * 3, mask)
        assert torch.allclose(loss_three, 3.0 * loss_one)

    def test_center_updates_in_mean_mode(self) -> None:
        torch.manual_seed(0)
        dims = [8, 16]
        loss_fn = FrancaIBOTPatchLoss(output_dims=dims, center_mode="mean")
        assert torch.all(loss_fn.centers[0].get_buffer("center") == 0)
        teacher, student, mask = _ibot_inputs(2, 4, 4, dims, num_masked_per_image=3)
        loss_fn(teacher, student, mask)
        assert not torch.all(loss_fn.centers[0].get_buffer("center") == 0)
        assert not torch.all(loss_fn.centers[1].get_buffer("center") == 0)

    def test_no_centers_in_sinkhorn_mode(self) -> None:
        loss_fn = FrancaIBOTPatchLoss(output_dims=[8, 16], center_mode="sinkhorn")
        assert len(loss_fn.centers) == 0

    @pytest.mark.parametrize(
        "make",
        [
            lambda: FrancaIBOTPatchLoss(output_dims=[]),
            lambda: FrancaIBOTPatchLoss(output_dims=[0, 8]),
            lambda: FrancaIBOTPatchLoss(output_dims=[-4, 8]),
            lambda: FrancaIBOTPatchLoss(output_dims=[8], center_mode="unknown"),
        ],
        ids=["empty", "zero-dim", "negative-dim", "bad-center-mode"],
    )
    def test_invalid_init(self, make: Callable[[], FrancaIBOTPatchLoss]) -> None:
        with pytest.raises(ValueError):
            make()

    def test_level_count_mismatch_raises(self) -> None:
        dims = [8, 16]
        loss_fn = FrancaIBOTPatchLoss(output_dims=dims)
        teacher, student, mask = _ibot_inputs(2, 4, 4, [8], num_masked_per_image=3)
        with pytest.raises(ValueError):
            loss_fn(teacher, student, mask)

    def test_empty_mask_raises(self) -> None:
        dims = [8, 16]
        loss_fn = FrancaIBOTPatchLoss(output_dims=dims)
        teacher = [torch.randn(0, dim) for dim in dims]
        student = [torch.randn(0, dim) for dim in dims]
        mask = torch.zeros(2, 4, 4, dtype=torch.bool)
        with pytest.raises(ValueError):
            loss_fn(teacher, student, mask)
