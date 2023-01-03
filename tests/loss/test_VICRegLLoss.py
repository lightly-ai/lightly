import unittest
import torch

from lightly.loss import VICRegLLoss

class TestVICRegLLoss(unittest.TestCase):
    
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
