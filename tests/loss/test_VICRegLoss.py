import unittest
import torch

from lightly.loss import VICRegLoss

class TestVICRegLoss(unittest.TestCase):
    def test_forward_pass(self):
        loss = VICRegLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 32))
            x1 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss(x0, x1)
            l2 = loss(x1, x0)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available")
    def test_forward_pass_cuda(self):
        loss = VICRegLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 32)).cuda()
            x1 = torch.randn((bsz, 32)).cuda()

            # symmetry
            l1 = loss(x0, x1)
            l2 = loss(x1, x0)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0)

    def test_forward_pass__error_batch_size_1(self):
        loss = VICRegLoss()
        x0 = torch.randn((1, 32))
        x1 = torch.randn((1, 32))
        with self.assertRaises(AssertionError):
            loss(x0, x1)

    def test_forward_pass__error_different_shapes(self):
        loss = VICRegLoss()
        x0 = torch.randn((2, 32))
        x1 = torch.randn((2, 16))
        with self.assertRaises(AssertionError):
            loss(x0, x1)
