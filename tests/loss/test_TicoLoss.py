import unittest
import torch

from lightly.loss.tico_loss import TiCoLoss
torch.manual_seed(0)
class TestVICRegLoss(unittest.TestCase):
    def test_forward_pass(self):
        loss = TiCoLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 32))
            x1 = torch.randn((bsz, 32))
            C =  torch.zeros((32, 32))

            # symmetry
            l1, C1 = loss(C, x0, x1)
            l2, C2 = loss(C, x1, x0)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0, 6)

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available")
    def test_forward_pass_cuda(self):
        loss = TiCoLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 32)).cuda()
            x1 = torch.randn((bsz, 32)).cuda()
            C =  torch.zeros((32, 32)).cuda()

            # symmetry
            l1, C1 = loss(C, x0, x1)
            l2, C2 = loss(C, x1, x0)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0, 4)

    def test_forward_pass__error_batch_size_1(self):
        loss = TiCoLoss()
        x0 = torch.randn((1, 32))
        x1 = torch.randn((1, 32))
        C = torch.zeros((32, 32))
        with self.assertRaises(AssertionError):
            loss(C, x0, x1)

    def test_forward_pass__error_different_shapes(self):
        loss = TiCoLoss()
        x0 = torch.randn((2, 32))
        x1 = torch.randn((2, 16))
        C = torch.zeros(8, 16)
        with self.assertRaises(AssertionError):
            loss(C, x0, x1)