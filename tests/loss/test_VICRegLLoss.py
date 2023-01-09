import unittest
import torch

from lightly.loss import VICRegLLoss

class TestVICRegLLoss(unittest.TestCase):

    def test_forward_pass(self):
        loss = VICRegLLoss()
        x0 = torch.randn((2, 32))
        x1 = torch.randn((2, 32))
        x0_L = torch.randn((2, 7, 7, 8))
        x1_L = torch.randn((2, 7, 7, 8))
        grid0 = torch.randn((2, 7, 7, 2))
        grid1 = torch.randn((2, 7, 7, 2))
        assert loss(x0, x1, x0_L, x1_L, grid0, grid1)

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available")
    def test_forward_pass_cuda(self):
        loss = VICRegLLoss()
        loss = VICRegLLoss()
        x0 = torch.randn((2, 32)).cuda()
        x1 = torch.randn((2, 32)).cuda()
        x0_L = torch.randn((2, 7, 7, 8)).cuda()
        x1_L = torch.randn((2, 7, 7, 8)).cuda()
        grid0 = torch.randn((2, 7, 7, 2)).cuda()
        grid1 = torch.randn((2, 7, 7, 2)).cuda()
        assert loss(x0, x1, x0_L, x1_L, grid0, grid1)

    
    def test_forward_pass__error_batch_size_1(self):
        loss = VICRegLLoss()
        x0 = torch.randn((1, 32))
        x1 = torch.randn((1, 32))
        x0_L = torch.randn((1, 7, 7, 8))
        x1_L = torch.randn((1, 7, 7, 8))
        grid0 = torch.randn((1, 7, 7, 2))
        grid1 = torch.randn((1, 7, 7, 2))
        with self.assertRaises(AssertionError):
            loss(x0, x1, x0_L, x1_L, grid0, grid1)

    def test_forward_pass__error_different_shapes(self):
        loss = VICRegLLoss()
        x0 = torch.randn((1, 32))
        x1 = torch.randn((1, 16))
        x0_L = torch.randn((1, 7, 7, 8))
        x1_L = torch.randn((1, 7, 7, 8))
        grid0 = torch.randn((1, 7, 7, 2))
        grid1 = torch.randn((1, 7, 7, 2))
        with self.assertRaises(AssertionError):
            loss(x0, x1, x0_L, x1_L, grid0, grid1)
