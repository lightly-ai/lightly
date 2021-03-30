import unittest
import torch

from lightly.loss.regularizer import CO2Regularizer

class TestCO2Regularizer(unittest.TestCase):

    def test_forward_pass_no_memory_bank(self):
        reg = CO2Regularizer(memory_bank_size=0)
        for bsz in range(1, 20):

            batch_1 = torch.randn((bsz, 32))
            batch_2 = torch.randn((bsz, 32))

            # symmetry
            l1 = reg(batch_1, batch_2)
            l2 = reg(batch_2, batch_1)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.)

    def test_forward_pass_memory_bank(self):
        reg = CO2Regularizer(memory_bank_size=4096)
        for bsz in range(1, 20):

            batch_1 = torch.randn((bsz, 32))
            batch_2 = torch.randn((bsz, 32))

            l1 = reg(batch_1, batch_2)
            self.assertGreater(l1.item(), 0)

    def test_forward_pass_cuda_no_memory_bank(self):
        if not torch.cuda.is_available():
            return

        reg = CO2Regularizer(memory_bank_size=0)
        for bsz in range(1, 20):

            batch_1 = torch.randn((bsz, 32)).cuda()
            batch_2 = torch.randn((bsz, 32)).cuda()

            # symmetry
            l1 = reg(batch_1, batch_2)
            l2 = reg(batch_2, batch_1)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.)


    def test_forward_pass_cuda_memory_bank(self):
        if not torch.cuda.is_available():
            return

        reg = CO2Regularizer(memory_bank_size=4096)
        for bsz in range(1, 20):

            batch_1 = torch.randn((bsz, 32)).cuda()
            batch_2 = torch.randn((bsz, 32)).cuda()

            # symmetry
            l1 = reg(batch_1, batch_2)
            self.assertGreater(l1.cpu().item(), 0)

