import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lightly.loss.koleo_loss import KoLeoLoss


class TestKoLeoLoss:
    # Test values generated using the original implementation from:
    # https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py
    @pytest.mark.parametrize(
        "x, expected_loss",
        [
            (torch.tensor([[1.0]]), 17.7275),
            (torch.tensor([[1.0, 1.0]]), 17.5393),
            (
                torch.tensor(
                    [
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [-1.0, 0.0],
                        [0.0, -1.0],
                        [-1.0, -1.0],
                    ]
                ),
                0.2674,
            ),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_forward(self, x: Tensor, expected_loss: float, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping test.")

        x = x.to(device)
        loss = KoLeoLoss().to(device)
        assert loss(x).item() == pytest.approx(expected_loss, rel=1e-4)
