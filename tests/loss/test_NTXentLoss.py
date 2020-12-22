import unittest
import torch

from lightly.loss import NTXentLoss


class TestNTXentLoss(unittest.TestCase):

    def test_get_correlated_mask(self):
        loss = NTXentLoss()
        for bsz in range(1, 100):
            mask = loss._torch_get_correlated_mask(bsz)

            # correct number of zeros in mask
            self.assertAlmostEqual(mask.sum(), 4*(bsz*bsz - bsz))

            # if mask is correct,
            # (1 - mask) * v adds up the first and second half of v
            v = torch.randn((2*bsz))
            mv = torch.mv(1. - mask.float(), v)
            vv = (v[bsz:] + v[:bsz]).repeat(2)
            self.assertAlmostEqual((mv - vv).pow(2).sum(), 0.)

    def test_forward_pass(self):
        loss = NTXentLoss(memory_bank_size=0)
        for bsz in range(1, 20):

            batch_1 = torch.randn((bsz, 32))
            batch_2 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.)

    def test_forward_pass_1d(self):
        loss = NTXentLoss(memory_bank_size=0)
        for bsz in range(1, 20):

            batch_1 = torch.randn((bsz, 1))
            batch_2 = torch.randn((bsz, 1))

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.)

    def test_forward_pass_neg_temp(self):
        loss = NTXentLoss(temperature=-1., memory_bank_size=0)
        for bsz in range(1, 20):

            batch_1 = torch.randn((bsz, 32))
            batch_2 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.)
    
    def test_forward_pass_memory_bank(self):
        loss = NTXentLoss(memory_bank_size=64)
        for bsz in range(1, 20):
            batch_1 = torch.randn((bsz, 32))
            batch_2 = torch.randn((bsz, 32))
            l = loss(batch_1, batch_2)

    def test_forward_pass_memory_bank_cuda(self):
        if not torch.cuda.is_available():
            return

        loss = NTXentLoss(memory_bank_size=64)
        for bsz in range(1, 20):
            batch_1 = torch.randn((bsz, 32)).cuda()
            batch_2 = torch.randn((bsz, 32)).cuda()
            l = loss(batch_1, batch_2)

    def test_forward_pass_cuda(self):
        if torch.cuda.is_available():
            loss = NTXentLoss(memory_bank_size=0)
            for bsz in range(1, 20):

                batch_1 = torch.randn((bsz, 32)).cuda()
                batch_2 = torch.randn((bsz, 32)).cuda()

                # symmetry
                l1 = loss(batch_1, batch_2)
                l2 = loss(batch_2, batch_1)
                self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.)
        else:
            pass

