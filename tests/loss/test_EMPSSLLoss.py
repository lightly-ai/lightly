import unittest

import torch

from lightly.loss.emp_ssl_loss import EMPSSLLoss


class testEMPSSSLLoss(unittest.TestCase):
    def test_forward(self) -> None:
        bs = 512
        dim = 128
        num_views = 100

        loss_fn = EMPSSLLoss()
        x = [torch.randn(bs, dim)] * num_views

        loss_fn(x)

    @unittest.skipUnless(torch.cuda.is_available(), "cuda not available")
    def test_forward_cuda(self) -> None:
        bs = 512
        dim = 128
        num_views = 100

        loss_fn = EMPSSLLoss().cuda()
        x = x = [torch.randn(bs, dim).cuda()] * num_views

        loss_fn(x)
