import torch
from torch import nn

from lightly.models.modules.heads import LeJEPAProjectionHead
from lightly.models.modules.lejepa import LeJEPAEncoder


def _make_encoder() -> LeJEPAEncoder:
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d(1),
    )
    projection_head = LeJEPAProjectionHead(input_dim=8, hidden_dim=16, output_dim=4)
    return LeJEPAEncoder(backbone=backbone, projection_head=projection_head)


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
