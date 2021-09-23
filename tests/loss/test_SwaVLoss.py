import unittest

import numpy as np
import torch

from lightly.loss import SwaVLoss


class TestNTXentLoss(unittest.TestCase):

    def test_forward_pass(self):

        n = 32
        n_high_res = 2
        high_res = [torch.eye(32, 32) for i in range(n_high_res)]

        for n_low_res in range(6):
            for sinkhorn_iterations in range(3):
                criterion = SwaVLoss(sinkhorn_iterations=sinkhorn_iterations)
                low_res = [torch.eye(n, n) for i in range(n_low_res)]
                
                with self.subTest(msg=f'n_low_res={n_low_res}, sinkhorn_iterations={sinkhorn_iterations}'):
                    loss = criterion(high_res, low_res)
                    # loss should be almost zero for unit matrix
                    self.assertGreater(0.5, loss.cpu().numpy())

    def test_forward_pass_bsz_1(self):

        n = 32
        n_high_res = 2
        high_res = [torch.eye(1, n) for i in range(n_high_res)]

        for n_low_res in range(6):
            for sinkhorn_iterations in range(3):
                criterion = SwaVLoss(sinkhorn_iterations=sinkhorn_iterations)
                low_res = [torch.eye(1, n) for i in range(n_low_res)]
                
                with self.subTest(msg=f'n_low_res={n_low_res}, sinkhorn_iterations={sinkhorn_iterations}'):
                    loss = criterion(high_res, low_res)

    def test_forward_pass_1d(self):
        n = 32
        n_high_res = 2
        high_res = [torch.eye(n, 1) for i in range(n_high_res)]

        for n_low_res in range(6):
            for sinkhorn_iterations in range(3):
                criterion = SwaVLoss(sinkhorn_iterations=sinkhorn_iterations)
                low_res = [torch.eye(n, 1) for i in range(n_low_res)]
                
                with self.subTest(msg=f'n_low_res={n_low_res}, sinkhorn_iterations={sinkhorn_iterations}'):
                    loss = criterion(high_res, low_res)
                    # loss should be almost zero for unit matrix
                    self.assertGreater(0.5, loss.cpu().numpy())

    @unittest.skipUnless(torch.cuda.is_available(), "skip")
    def test_forward_pass_cuda(self):
        n = 32
        n_high_res = 2
        high_res = [torch.eye(n, n).cuda() for i in range(n_high_res)]

        for n_low_res in range(6):
            for sinkhorn_iterations in range(3):
                criterion = SwaVLoss(sinkhorn_iterations=sinkhorn_iterations)
                low_res = [torch.eye(n, n).cuda() for i in range(n_low_res)]
                
                with self.subTest(msg=f'n_low_res={n_low_res}, sinkhorn_iterations={sinkhorn_iterations}'):
                    loss = criterion(high_res, low_res)
                    # loss should be almost zero for unit matrix
                    self.assertGreater(0.5, loss.cpu().numpy())
