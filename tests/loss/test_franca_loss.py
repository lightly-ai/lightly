from __future__ import annotations

from typing import Callable

import pytest
import torch

from lightly.loss import DINOLoss
from lightly.loss.franca_loss import FrancaDINOLoss


def _views(
    num_views: int, batch_size: int, dims: list[int]
) -> list[tuple[torch.Tensor, ...]]:
    """Builds ``num_views`` views, each a tuple with one tensor per nesting level."""
    return [
        tuple(torch.randn(batch_size, dim) for dim in dims) for _ in range(num_views)
    ]


class TestFrancaDINOLoss:
    def test_forward_and_backward__mean(self) -> None:
        torch.manual_seed(0)
        dims = [8, 16, 24]
        loss_fn = FrancaDINOLoss(output_dims=dims, center_mode="mean")
        teacher = _views(num_views=2, batch_size=4, dims=dims)
        student = [
            tuple(s.requires_grad_(True) for s in view) for view in _views(2, 4, dims)
        ]
        loss = loss_fn(teacher, student)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        for view in student:
            for tensor in view:
                assert tensor.grad is not None

    def test_forward_and_backward__sinkhorn(self) -> None:
        torch.manual_seed(0)
        dims = [8, 16, 24]
        loss_fn = FrancaDINOLoss(output_dims=dims, center_mode="sinkhorn")
        teacher = _views(num_views=2, batch_size=4, dims=dims)
        student = [
            tuple(s.requires_grad_(True) for s in view) for view in _views(2, 4, dims)
        ]
        loss = loss_fn(teacher, student)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        for view in student:
            for tensor in view:
                assert tensor.grad is not None

    def test_more_student_than_teacher_views(self) -> None:
        """Multi-crop path: 2 teacher (global) views and more student views."""
        torch.manual_seed(0)
        dims = [8, 16]
        loss_fn = FrancaDINOLoss(output_dims=dims, center_mode="mean")
        teacher = _views(num_views=2, batch_size=3, dims=dims)
        student = [
            tuple(s.requires_grad_(True) for s in view) for view in _views(6, 3, dims)
        ]
        loss = loss_fn(teacher, student)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        for view in student:
            for tensor in view:
                assert tensor.grad is not None

    def test_single_level_matches_dino_loss(self) -> None:
        """A single-level FrancaDINOLoss (mean) reduces exactly to lightly's DINOLoss."""
        torch.manual_seed(0)
        dim = 16
        teacher = [torch.randn(4, dim), torch.randn(4, dim)]
        student = [torch.randn(4, dim), torch.randn(4, dim)]

        dino = DINOLoss(output_dim=dim)
        franca = FrancaDINOLoss(output_dims=[dim], center_mode="mean")

        loss_dino = dino(teacher, student)
        loss_franca = franca([(t,) for t in teacher], [(s,) for s in student])

        assert torch.allclose(loss_dino, loss_franca)
        assert torch.allclose(dino.center, franca.get_buffer("center_0"))

    def test_levels_are_summed_unweighted(self) -> None:
        """With equal per-level inputs and zero centers, the total is the level sum."""
        torch.manual_seed(0)
        dim = 12
        teacher_level = [torch.randn(4, dim), torch.randn(4, dim)]
        student_level = [torch.randn(4, dim), torch.randn(4, dim)]

        one = FrancaDINOLoss(output_dims=[dim], center_mode="mean")
        three = FrancaDINOLoss(output_dims=[dim, dim, dim], center_mode="mean")

        loss_one = one([(t,) for t in teacher_level], [(s,) for s in student_level])
        loss_three = three(
            [(t, t, t) for t in teacher_level], [(s, s, s) for s in student_level]
        )
        assert torch.allclose(loss_three, 3.0 * loss_one)

    def test_center_updates_in_mean_mode(self) -> None:
        torch.manual_seed(0)
        dims = [8, 16]
        loss_fn = FrancaDINOLoss(output_dims=dims, center_mode="mean")
        assert torch.all(loss_fn.get_buffer("center_0") == 0)
        assert torch.all(loss_fn.get_buffer("center_1") == 0)
        teacher = _views(num_views=2, batch_size=4, dims=dims)
        student = _views(num_views=2, batch_size=4, dims=dims)
        loss_fn(teacher, student)
        assert not torch.all(loss_fn.get_buffer("center_0") == 0)
        assert not torch.all(loss_fn.get_buffer("center_1") == 0)

    def test_center_unused_in_sinkhorn_mode(self) -> None:
        torch.manual_seed(0)
        dims = [8, 16]
        loss_fn = FrancaDINOLoss(output_dims=dims, center_mode="sinkhorn")
        teacher = _views(num_views=2, batch_size=4, dims=dims)
        student = _views(num_views=2, batch_size=4, dims=dims)
        loss_fn(teacher, student)
        # Sinkhorn mode does not maintain a running center.
        assert torch.all(loss_fn.get_buffer("center_0") == 0)
        assert torch.all(loss_fn.get_buffer("center_1") == 0)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: FrancaDINOLoss(output_dims=[]),
            lambda: FrancaDINOLoss(output_dims=[0, 8]),
            lambda: FrancaDINOLoss(output_dims=[-4, 8]),
            lambda: FrancaDINOLoss(output_dims=[8], center_mode="unknown"),
        ],
        ids=["empty", "zero-dim", "negative-dim", "bad-center-mode"],
    )
    def test_invalid_init(self, make: Callable[[], FrancaDINOLoss]) -> None:
        with pytest.raises(ValueError):
            make()

    def test_view_level_count_mismatch_raises(self) -> None:
        dims = [8, 16]
        loss_fn = FrancaDINOLoss(output_dims=dims)
        teacher = _views(num_views=2, batch_size=4, dims=dims)
        # A student view exposes only one level while the loss expects two.
        student = [(torch.randn(4, 8),), (torch.randn(4, 8),)]
        with pytest.raises(ValueError):
            loss_fn(teacher, student)
