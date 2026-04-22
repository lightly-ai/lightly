import pytest
import torch
from pytest_mock import MockerFixture
from torch import distributed as dist
from torch import nn

from lightly.models.modules.heads import LeJEPAProjectionHead
from lightly.models.modules.lejepa import (
    LeJEPAEncoder,
    LeJEPALoss,
    lejepa_invariance_loss,
)


def _make_encoder() -> LeJEPAEncoder:
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d(1),
    )
    projection_head = LeJEPAProjectionHead(input_dim=8, hidden_dim=16, output_dim=4)
    return LeJEPAEncoder(backbone=backbone, projection_head=projection_head)


class TestLeJEPAInvarianceLoss:
    def test_forward(self) -> None:
        torch.manual_seed(0)
        proj = torch.randn(8, 32, 128)
        loss = lejepa_invariance_loss(proj)
        assert loss.isfinite()
        assert loss.ndim == 0

    def test_backward(self) -> None:
        torch.manual_seed(0)
        proj = torch.randn(8, 32, 128, requires_grad=True)
        loss = lejepa_invariance_loss(proj)
        loss.backward()
        assert proj.grad is not None
        assert proj.grad.shape == proj.shape


class TestLeJEPALoss:
    def test_backward_pass(self) -> None:
        torch.manual_seed(0)
        loss_fn = LeJEPALoss()
        proj = torch.randn(8, 32, 128, requires_grad=True)
        loss = loss_fn(proj)
        loss.backward()
        assert proj.grad is not None
        assert proj.grad.shape == proj.shape

    def test_forward(self) -> None:
        torch.manual_seed(0)
        loss_fn = LeJEPALoss()
        proj = torch.randn(8, 32, 128)
        loss = loss_fn(proj)
        assert loss.isfinite()

    def test_forward_gather_distributed_world_size_gt_one(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch("lightly.loss.lejepa_loss.lightly_dist.world_size", return_value=2)
        mock_broadcast = mocker.patch.object(dist, "broadcast")
        mock_all_reduce = mocker.patch.object(dist, "all_reduce")

        torch.manual_seed(0)
        loss_fn = LeJEPALoss(gather_distributed=True)
        proj = torch.randn(8, 32, 128)
        loss = loss_fn(proj)

        assert loss.isfinite()
        mock_broadcast.assert_called_once()
        assert mock_all_reduce.call_count == 3

    @pytest.mark.parametrize("lambda_param", [-0.1, 1.1])
    def test_lambda_must_be_in_unit_interval(self, lambda_param: float) -> None:
        with pytest.raises(ValueError):
            LeJEPALoss(lambda_param=lambda_param)


class TestLeJEPAEncoder:
    def test_forward_shape(self) -> None:
        torch.manual_seed(0)
        encoder = _make_encoder()
        x = torch.randn(2, 3, 32, 32)
        y = encoder(x)
        assert y.shape == (2, 4)

    def test_backward(self) -> None:
        torch.manual_seed(0)
        encoder = _make_encoder()
        x = torch.randn(2, 3, 32, 32)
        y = encoder(x)
        y.sum().backward()
        assert any(p.grad is not None for p in encoder.backbone.parameters())
        assert any(p.grad is not None for p in encoder.projection_head.parameters())
