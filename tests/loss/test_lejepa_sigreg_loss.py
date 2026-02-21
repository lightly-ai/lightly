import pytest
import torch

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

    def test_deterministic_output(self) -> None:
        torch.manual_seed(0)
        loss_fn = SIGReg()
        proj = torch.randn(10, 1024, 16)
        loss1 = loss_fn(proj)

        torch.manual_seed(0)
        proj = torch.randn(10, 1024, 16)
        loss2 = loss_fn(proj)
        assert torch.isclose(loss1, loss2)

    @pytest.mark.parametrize("channel_size", [8, 16, 32, 64])
    def test_works_with_any_channel_size(self, channel_size: int) -> None:
        torch.manual_seed(0)
        loss_fn = SIGReg()
        proj = torch.randn(10, 1024, channel_size)
        loss = loss_fn(proj)
        assert loss.item() >= 0.0

    @pytest.mark.parametrize("batch_size", [1, 4, 10, 32])
    def test_works_with_any_batch_size(self, batch_size: int) -> None:
        torch.manual_seed(0)
        loss_fn = SIGReg()
        proj = torch.randn(batch_size, 1024, 16)
        loss = loss_fn(proj)
        assert loss.item() >= 0.0

    def test_dont_work_with_1d_input(self) -> None:
        torch.manual_seed(0)
        loss_fn = SIGReg()
        proj = torch.randn(1024)
        with pytest.raises(IndexError):
            loss_fn(proj)

    @pytest.mark.parametrize("knots", [1, 0, -1, -5])
    def test_knots_must_be_greater_than_one(self, knots: int) -> None:
        with pytest.raises(ValueError):
            SIGReg(knots=knots)

    def test_works_with_non_float32_input(self) -> None:
        torch.manual_seed(0)
        loss_fn = SIGReg()
        proj = torch.randn(10, 1024, 16, dtype=torch.float64)
        loss = loss_fn(proj)
        assert loss.item() >= 0.0

    def test_non_gaussian_positive(self) -> None:
        torch.manual_seed(0)
        loss_fn = SIGReg()
        proj = torch.randn(10, 1024, 16)
        loss_gaussian = loss_fn(proj)
        proj_skewed = proj * 1.5 + 0.3
        loss_skewed = loss_fn(proj_skewed)
        assert loss_skewed.item() > 0.0
        assert loss_skewed.item() > loss_gaussian.item()
