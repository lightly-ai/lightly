import pytest
import torch
from pytest_mock import MockerFixture
from torch import distributed as dist

from lightly.loss.lejepa_loss import SIGReg


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

    def test_forward_non_float32_input(self) -> None:
        torch.manual_seed(0)
        loss_fn = SIGReg()
        proj = torch.randn(10, 1024, 16, dtype=torch.float16)
        loss = loss_fn(proj)
        assert loss.isfinite()

    def test_knots_must_be_greater_than_one(self) -> None:
        with pytest.raises(ValueError):
            SIGReg(knots=1)

    def test_t_max_must_be_greater_than_zero(self) -> None:
        with pytest.raises(ValueError):
            SIGReg(t_max=0.0)

    def test_num_vectors_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            SIGReg(num_vectors=0)
