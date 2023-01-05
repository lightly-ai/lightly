import unittest
import torch

from lightly.loss.tico_loss import TiCoLoss

class TestTiCoLoss(unittest.TestCase):
    def test_forward_pass(self):
        torch.manual_seed(0)
        loss = TiCoLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 256))
            x1 = torch.randn((bsz, 256))

            # symmetry
            l1 = loss(x0, x1, update_covariance_matrix=False)
            l2 = loss(x1, x0, update_covariance_matrix=False)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available")
    def test_forward_pass_cuda(self):
        torch.manual_seed(0)
        loss = TiCoLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 256)).cuda()
            x1 = torch.randn((bsz, 256)).cuda()

            # symmetry
            l1 = loss(x0, x1, update_covariance_matrix=False)
            l2 = loss(x1, x0, update_covariance_matrix=False)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0, 2)

    def test_forward_pass__error_batch_size_1(self):
        torch.manual_seed(0)
        loss = TiCoLoss()
        x0 = torch.randn((1, 256))
        x1 = torch.randn((1, 256))
        with self.assertRaises(AssertionError):
            loss(x0, x1, update_covariance_matrix=False)

    def test_forward_pass__error_different_shapes(self):
        torch.manual_seed(0)
        loss = TiCoLoss()
        x0 = torch.randn((2, 32))
        x1 = torch.randn((2, 16))
        with self.assertRaises(AssertionError):
            loss(x0, x1, update_covariance_matrix=False)