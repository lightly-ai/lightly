import unittest

import numpy as np
import torch

from lightly.loss import NTXentLoss


class TestNTXentLoss(unittest.TestCase):

    def test_with_values(self):
        out0 = [[1., 1., 1.], [-1., -1., -1.]]
        out1 = [[1., 1., -1.], [-1., -1., 1.]]
        out0 = torch.FloatTensor(out0)
        out1 = torch.FloatTensor(out1)
        sim_matrix = [[]]

        loss_function = NTXentLoss()
        l1 = loss_function(out0, out1)
        l2 = loss_function(out1, out0)
        self.assertAlmostEqual(float(l1), 0.0625955, places=5)
        self.assertAlmostEqual(float(l2), 0.0625955, places=5)

    def calc_loss_manual(self, out0: np.ndarray, out1: np.ndarray, temperature: float) -> loss:
        # using the pseudocode directly from https://arxiv.org/pdf/2002.05709.pdf Algorithm1
        z = np.concatenate([out0, out1], axis=1)
        N = len(out0)

        s_i_j = np.zeros((2 * len(out0), 2 * len(out1)))
        for i in range(2 * N):
            for j in range(2 * N):
                s_i_j[i, j] = z[i].T * z[j] / (np.linalg.norm(z[i]) * np.linalg.norm(z[j]))

        logit_i_j = np.exp(s_i_j / temperature)

        l_i_j = np.zeros_like(logit_i_j)
        for i in range(2 * N):
            for j in range(2 * N):
                nominator = logit_i_j[i, j]
                denominator = 0
                for k in range(2 * N):
                    if k != i:
                        denominator += logit_i_j[i, k]
                l_i_j[i, j] = -1 * np.log(nominator/denominator)

        loss = 0
        for k in range(N):
            loss += l_i_j[2*k-1, 2*k] + l_i_j[2*k, 2*k-1]
        loss /= (2*N)
        return loss

    def test_with_values_and_memory_bank(self):
        out0 = [[1., 1., 1.], [-1., -1., -1.]]
        out1 = [[1., 1., 0.], [-1., -1., 0.]]
        out0 = torch.FloatTensor(out0)
        out1 = torch.FloatTensor(out1)

        torch.manual_seed(42)
        loss = NTXentLoss(memory_bank_size=64)
        l1 = loss(out0, out1)
        l2 = loss(out1, out0)
        for l in [l1, l2]:
            self.assertAlmostEqual(float(l), 4.5, delta=0.5)

    def test_get_correlated_mask(self):
        loss = NTXentLoss()
        for bsz in range(1, 100):
            mask = loss._torch_get_correlated_mask(bsz)

            # correct number of zeros in mask
            self.assertAlmostEqual(mask.sum(), 4 * (bsz * bsz - bsz))

            # if mask is correct,
            # (1 - mask) * v adds up the first and second half of v
            v = torch.randn((2 * bsz))
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
