import pytest
import torch

from lightly.loss.mmcr_loss import MMCRLoss


class TestMMCRLoss:
    def test_forward(self) -> None:
        bs = 3
        dim = 128
        k = 32

        loss_fn = MMCRLoss()
        online = torch.randn(bs, k, dim)
        momentum = torch.randn(bs, k, dim)

        loss_fn(online, momentum)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_forward_cuda(self) -> None:
        bs = 3
        dim = 128
        k = 32

        loss_fn = MMCRLoss()
        online = torch.randn(bs, k, dim).cuda()
        momentum = torch.randn(bs, k, dim).cuda()

        loss_fn(online, momentum)

    def test_loss_value(self) -> None:
        """If all values are zero, the loss should be zero."""
        bs = 3
        dim = 128
        k = 32

        loss_fn = MMCRLoss()
        online = torch.zeros(bs, k, dim)
        momentum = torch.zeros(bs, k, dim)

        loss = loss_fn(online, momentum)
        assert loss == 0.0

    def test_lambda_value_error(self) -> None:
        """If lambda is negative, a ValueError should be raised."""
        with pytest.raises(ValueError):
            MMCRLoss(lmda=-1)

    def test_shape_assertion_forward(self) -> None:
        bs = 3
        dim = 128
        k = 32

        loss_fn = MMCRLoss()
        online = torch.randn(bs, k, dim)
        momentum = torch.randn(bs, k, dim + 1)

        with pytest.raises(AssertionError):
            loss_fn(online, momentum)
