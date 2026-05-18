from typing import Tuple

import pytest
import torch
from pytest_mock import MockerFixture
from torch import distributed as dist

from lightly.loss import LeJEPALoss, SIGReg
from lightly.loss.lejepa_loss import lejepa_invariance_loss


class TestSIGReg:
    def test_backward_pass(self) -> None:
        torch.manual_seed(0)
        loss_fn = SIGReg()
        proj = torch.randn(10, 1024, 16, requires_grad=True)
        loss = loss_fn(proj)
        loss.backward()
        assert proj.grad is not None
        assert proj.grad.shape == proj.shape

    def test_forward(self) -> None:
        torch.manual_seed(0)
        loss_fn = SIGReg(t_max=1.5, num_vectors=32)
        proj = torch.randn(10, 1024, 16)
        loss_fn(proj)

    def test_forward_gather_distributed(self) -> None:
        torch.manual_seed(0)
        loss_fn = SIGReg(gather_distributed=True)
        proj = torch.randn(10, 1024, 16)
        loss = loss_fn(proj)
        assert loss.isfinite()

    def test_forward_gather_distributed_world_size_gt_one(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch("lightly.loss.lejepa_loss.lightly_dist.world_size", return_value=2)
        mock_broadcast = mocker.patch.object(dist, "broadcast")
        mock_all_reduce = mocker.patch.object(dist, "all_reduce")

        torch.manual_seed(0)
        loss_fn = SIGReg(gather_distributed=True)
        proj = torch.randn(10, 1024, 16)
        loss = loss_fn(proj)

        assert loss.isfinite()
        mock_broadcast.assert_called_once()
        assert mock_all_reduce.call_count == 3

    def test_knots_must_be_greater_than_one(self) -> None:
        with pytest.raises(ValueError):
            SIGReg(knots=1)

    def test_t_max_must_be_greater_than_zero(self) -> None:
        with pytest.raises(ValueError):
            SIGReg(t_max=0.0)

    def test_num_vectors_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            SIGReg(num_vectors=0)


class TestLeJEPAInvarianceLoss:
    def test_forward(self) -> None:
        torch.manual_seed(0)
        local_proj = torch.randn(6, 32, 128)
        global_proj = torch.randn(2, 32, 128)
        loss = lejepa_invariance_loss(local_proj=local_proj, global_proj=global_proj)
        assert loss.isfinite()
        assert loss.ndim == 0

    def test_backward(self) -> None:
        torch.manual_seed(0)
        local_proj = torch.randn(8, 32, 128, requires_grad=True)
        global_proj = torch.randn(8, 32, 128, requires_grad=True)
        loss = lejepa_invariance_loss(local_proj=local_proj, global_proj=global_proj)
        loss.backward()
        assert local_proj.grad is not None
        assert local_proj.grad.shape == local_proj.shape
        assert global_proj.grad is not None
        assert global_proj.grad.shape == global_proj.shape

    @pytest.mark.parametrize(
        ("local_shape", "global_shape"),
        [
            ((32, 128), (8, 32, 128)),
            ((8, 32, 128), (32, 128)),
            ((8, 16, 128), (8, 32, 128)),
            ((8, 32, 64), (8, 32, 128)),
        ],
    )
    def test_validates_projection_shapes(
        self, local_shape: Tuple[int, ...], global_shape: Tuple[int, ...]
    ) -> None:
        local_proj = torch.randn(*local_shape)
        global_proj = torch.randn(*global_shape)

        with pytest.raises(ValueError):
            lejepa_invariance_loss(local_proj=local_proj, global_proj=global_proj)


class TestLeJEPALoss:
    def test_backward_pass(self) -> None:
        torch.manual_seed(0)
        loss_fn = LeJEPALoss()
        local_proj = torch.randn(6, 32, 128, requires_grad=True)
        global_proj = torch.randn(2, 32, 128, requires_grad=True)
        loss = loss_fn(local_proj=local_proj, global_proj=global_proj)
        loss.backward()
        assert local_proj.grad is not None
        assert local_proj.grad.shape == local_proj.shape
        assert global_proj.grad is not None
        assert global_proj.grad.shape == global_proj.shape

    def test_forward(self) -> None:
        torch.manual_seed(0)
        loss_fn = LeJEPALoss()
        local_proj = torch.randn(6, 32, 128)
        global_proj = torch.randn(2, 32, 128)
        loss = loss_fn(local_proj=local_proj, global_proj=global_proj)
        assert loss.isfinite()

    def test_forward_gather_distributed_world_size_gt_one(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch("lightly.loss.lejepa_loss.lightly_dist.world_size", return_value=2)
        mock_broadcast = mocker.patch.object(dist, "broadcast")
        mock_all_reduce = mocker.patch.object(dist, "all_reduce")

        torch.manual_seed(0)
        loss_fn = LeJEPALoss(gather_distributed=True)
        local_proj = torch.randn(8, 32, 128)
        global_proj = torch.randn(8, 32, 128)
        loss = loss_fn(local_proj=local_proj, global_proj=global_proj)

        assert loss.isfinite()
        mock_broadcast.assert_called_once()
        assert mock_all_reduce.call_count == 3

    @pytest.mark.parametrize("lambda_param", [-0.1, 1.1])
    def test_lambda_must_be_in_unit_interval(self, lambda_param: float) -> None:
        with pytest.raises(ValueError):
            LeJEPALoss(lambda_param=lambda_param)

    def test_lambda_zero_equals_pure_invariance(self) -> None:
        # Wiring check: at lambda=0 the SIGReg term is zeroed out, so the loss
        # must equal the standalone invariance loss on the same projections.
        torch.manual_seed(0)
        local_proj = torch.randn(8, 32, 128)
        global_proj = torch.randn(8, 32, 128)

        lejepa_loss = LeJEPALoss(lambda_param=0.0)(
            local_proj=local_proj, global_proj=global_proj
        )
        invariance_only = lejepa_invariance_loss(
            local_proj=local_proj, global_proj=global_proj
        )

        assert torch.allclose(lejepa_loss, invariance_only)

    def test_lambda_one_equals_pure_sigreg(self) -> None:
        # Wiring check: at lambda=1 the invariance term is zeroed out, so the
        # loss must equal a standalone SIGReg call under the same RNG state.
        lejepa_fn = LeJEPALoss(lambda_param=1.0)
        sigreg_fn = SIGReg()
        torch.manual_seed(0)
        local_proj = torch.randn(8, 32, 128)
        global_proj = torch.randn(8, 32, 128)

        torch.manual_seed(42)
        lejepa_loss = lejepa_fn(local_proj=local_proj, global_proj=global_proj)

        torch.manual_seed(42)
        sigreg_loss = sigreg_fn(local_proj)

        assert torch.allclose(lejepa_loss, sigreg_loss)
