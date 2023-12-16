import unittest

import torch

from lightly.loss.wmse_loss import WMSELoss


class testWMSELoss(unittest.TestCase):
    def test_forward(self) -> None:
        bs = 128
        dim = 64
        num_samples = 32

        loss_fn = WMSELoss()
        x = torch.randn(bs * num_samples, dim)

        loss = loss_fn(x)

        print(loss)

    @unittest.skipUnless(torch.cuda.is_available(), "cuda not available")
    def test_forward_cuda(self) -> None:
        bs = 128
        dim = 64
        num_samples = 32

        loss_fn = WMSELoss().cuda()
        x = torch.randn(bs * num_samples, dim).cuda()

        loss = loss_fn(x)

        print(loss)

    def test_loss_value(self) -> None:
        """If all values are zero, the loss should be zero."""
        bs = 128
        dim = 64
        num_samples = 32

        loss_fn = WMSELoss()
        x = torch.randn(bs * num_samples, dim)

        loss = loss_fn(x)

        self.assertGreater(loss, 0)

    def test_num_samples_error(self) -> None:
        with self.assertRaises(RuntimeError):
            loss_fn = WMSELoss(num_samples=3)
            x = torch.randn(5, 64)
            loss_fn(x)

    def test_w_size_error(self) -> None:
        with self.assertRaises(ValueError):
            loss_fn = WMSELoss(w_size=5)
            x = torch.randn(4, 64)
            loss_fn(x)
