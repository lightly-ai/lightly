import pytest
import torch

from lightly.loss.lejepa_loss import SIGReg


class TestSIGReg:
    def test_non_gaussian_positive(self) -> None:
        torch.manual_seed(0)
        loss_fn = SIGReg()
        proj = torch.randn(10, 1024, 16)
        loss_gaussian = loss_fn(proj)
        proj_skewed = proj * 1.5 + 0.3
        loss_skewed = loss_fn(proj_skewed)
        print(f"{loss_gaussian.item()=}, {loss_skewed.item()=}")
        breakpoint()
        assert loss_skewed.item() > 0.0
        assert loss_skewed.item() > loss_gaussian.item()
