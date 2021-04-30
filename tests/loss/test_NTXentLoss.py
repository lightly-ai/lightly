import unittest

import numpy as np
import torch

from lightly.loss import NTXentLoss


class TestNTXentLoss(unittest.TestCase):

    def test_with_values(self):
        for n_samples in [1, 2, 4]:
            for dimension in [1, 2, 16, 64]:
                for temperature in [0.1, 1, 10]:
                    out0 = np.random.normal(0, 1, size=(n_samples, dimension))
                    out1 = np.random.normal(0, 1, size=(n_samples, dimension))
                    with self.subTest(msg=f"out0.shape={out0.shape}, temperature={temperature}"):
                        out0 = torch.FloatTensor(out0)
                        out1 = torch.FloatTensor(out1)

                        loss_function = NTXentLoss(temperature=temperature)
                        l1 = float(loss_function(out0, out1))
                        l2 = float(loss_function(out1, out0))
                        l1_manual = self.calc_ntxent_loss_manual(out0, out1, temperature=temperature)
                        l2_manual = self.calc_ntxent_loss_manual(out0, out1, temperature=temperature)
                        self.assertAlmostEqual(l1, l2, places=5)
                        self.assertAlmostEqual(l1, l1_manual, places=5)
                        self.assertAlmostEqual(l2, l2_manual, places=5)

    def calc_ntxent_loss_manual(self, out0, out1, temperature: float) -> float:
        # using the pseudocode directly from https://arxiv.org/pdf/2002.05709.pdf , Algorithm 1

        out0 = np.array(out0)
        out1 = np.array(out1)

        N = len(out0)
        z = np.concatenate([out0, out1], axis=0)
        # different to the notation in the paper, in our case z[k] and z[k+N]
        # are different augmentations of the same image

        s_i_j = np.zeros((2 * len(out0), 2 * len(out1)))
        for i in range(2 * N):
            for j in range(2 * N):
                sim = np.inner(z[i], z[j]) / (np.linalg.norm(z[i]) * np.linalg.norm(z[j]))
                s_i_j[i, j] = sim

        exponential_i_j = np.exp(s_i_j / temperature)

        l_i_j = np.zeros_like(exponential_i_j)
        for i in range(2 * N):
            for j in range(2 * N):
                nominator = exponential_i_j[i, j]
                denominator = 0
                for k in range(2 * N):
                    if k != i:
                        denominator += exponential_i_j[i, k]
                l_i_j[i, j] = -1 * np.log(nominator / denominator)

        loss = 0
        for k in range(N):
            loss += l_i_j[k, k + N] + l_i_j[k + N, k]
        loss /= (2 * N)
        return loss

    def test_with_correlated_embedding(self):
        n_samples = 8
        temperature = 3
        memory_bank_size = n_samples
        out0 = np.random.random((n_samples, 1))
        out1 = np.random.random((n_samples, 1))
        out0 = np.concatenate([out0, 2 * out0], axis=1)
        out1 = np.concatenate([out1, 2 * out1], axis=1)
        out0 = torch.FloatTensor(out0)
        out1 = torch.FloatTensor(out1)
        out0.requires_grad = True

        loss_function = NTXentLoss(temperature=temperature)
        loss = float(loss_function(out0, out1))
        expected_loss = -1 * np.log(1 / (2 * n_samples - 1))
        loss_function_with_bank = NTXentLoss(temperature=temperature, memory_bank_size=memory_bank_size)
        for i in range(2):
            loss_with_memory_bank = float(loss_function_with_bank(out0, out1))
        self.assertAlmostEqual(loss, expected_loss, places=1)


        expected_loss_memory_bank = -1 * np.log(1 / (memory_bank_size+1))
        self.assertAlmostEqual(loss_with_memory_bank, expected_loss_memory_bank, places=3)

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
            mask = loss._torch_get_mask_negative_samples(bsz)

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
