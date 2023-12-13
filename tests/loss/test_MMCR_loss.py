import unittest

import torch

from lightly.loss.mmcr_loss import MMCRLoss


class testMMCRLoss(unittest.TestCase):
    def test_forward(self) -> None:
        bs = 3
        dim = 128
        k = 32

        loss_fn = MMCRLoss()
        online = torch.randn(bs, k, dim)
        momentum = torch.randn(bs, k, dim)

        loss = loss_fn(online, momentum)

        print(loss)

    @unittest.skipUnless(torch.cuda.is_available(), "cuda not available")
    def test_forward_cuda(self) -> None:
        bs = 3
        dim = 128
        k = 32

        loss_fn = MMCRLoss()
        online = torch.randn(bs, k, dim).cuda()
        momentum = torch.randn(bs, k, dim).cuda()

        loss = loss_fn(online, momentum)

        print(loss)

    def test_loss_value(self) -> None:
        """If all values are zero, the loss should be zero."""
        bs = 3
        dim = 128
        k = 32

        loss_fn = MMCRLoss()
        online = torch.zeros(bs, k, dim)
        momentum = torch.zeros(bs, k, dim)

        loss = loss_fn(online, momentum)

        self.assertTrue(loss == 0)

    def test_lambda_value_error(self) -> None:
        """If lambda is negative, a ValueError should be raised."""
        with self.assertRaises(ValueError):
            MMCRLoss(lmda=-1)

    def test_shape_assertion_forward(self) -> None:
        bs = 3
        dim = 128
        k = 32

        loss_fn = MMCRLoss()
        online = torch.randn(bs, k, dim)
        momentum = torch.randn(bs, k, dim + 1)

        with self.assertRaises(AssertionError):
            loss_fn(online, momentum)
