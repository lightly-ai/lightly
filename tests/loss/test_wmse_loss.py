import pytest
import torch

from lightly.loss.wmse_loss import WMSELoss

try:
    import torch.linalg.solve_triangular
except ImportError:
    pytest.skip("torch.linalg.solve_triangular not available", allow_module_level=True)


class TestWMSELoss:
    def test_forward(self) -> None:
        bs = 512
        dim = 128
        num_samples = 2

        loss_fn = WMSELoss()
        x = torch.randn(bs * num_samples, dim)

        loss_fn(x)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_forward_cuda(self) -> None:
        bs = 512
        dim = 128
        num_samples = 2

        loss_fn = WMSELoss().cuda()
        x = torch.randn(bs * num_samples, dim).cuda()

        loss_fn(x)

    def test_loss_value(self) -> None:
        """If all values are zero, the loss should be zero."""
        bs = 512
        dim = 128
        num_samples = 2

        loss_fn = WMSELoss()
        x = torch.randn(bs * num_samples, dim)

        loss = loss_fn(x)
        assert loss > 0

    def test_embedding_dim_error(self) -> None:
        with pytest.raises(ValueError):
            WMSELoss(embedding_dim=2, w_size=2)

    def test_num_samples_error(self) -> None:
        with pytest.raises(RuntimeError):
            loss_fn = WMSELoss(num_samples=3)
            x = torch.randn(5, 128)
            loss_fn(x)

    def test_w_size_error(self) -> None:
        with pytest.raises(ValueError):
            loss_fn = WMSELoss(w_size=5)
            x = torch.randn(4, 128)
            loss_fn(x)
