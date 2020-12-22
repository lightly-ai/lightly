import unittest
import torch

from lightly.loss.memory_bank import MemoryBankModule


class TestNTXentLoss(unittest.TestCase):

    def test_NegativeSize(self):
        with self.assertRaises(ValueError):
            MemoryBankModule(size=-1)

    def test_ForwardEasy(self):
        bsz = 3
        dim, size = 2, 9
        n = 33 * bsz
        memory_bank = MemoryBankModule(size=size)

        ptr = 0
        for i in range(0, n, bsz):

            output = torch.randn(2 * bsz, dim)
            output.requires_grad = True
            out0, out1 = output[:bsz], output[bsz:]

            _, curr_memory_bank = memory_bank(out1)
            next_memory_bank = memory_bank.bank

            curr_diff = out0.T - curr_memory_bank[:, ptr:ptr + bsz]
            next_diff = out1.T - next_memory_bank[:, ptr:ptr + bsz]

            # the current memory bank should not hold the batch yet
            self.assertGreater(curr_diff.norm(), 1e-5)
            # the "next" memory bank should hold the batch
            self.assertGreater(1e-5, next_diff.norm())

            ptr = (ptr + bsz) % size

    def test_Forward(self):
        bsz = 3
        dim, size = 2, 10
        n = 33 * bsz
        memory_bank = MemoryBankModule(size=size)

        for i in range(0, n, bsz):

            # see if there are any problems when the bank size
            # is no multiple of the batch size
            output = torch.randn(bsz, dim)
            _, _ = memory_bank(output)
