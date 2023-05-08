import unittest

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture
from torch import distributed as dist

from lightly.loss import NTXentLoss


class TestNTXentLoss:
    def test__gather_distributed(self, mocker: MockerFixture) -> None:
        mock_is_available = mocker.patch.object(dist, "is_available", return_value=True)
        NTXentLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__gather_distributed_dist_not_available(
        self, mocker: MockerFixture
    ) -> None:
        mock_is_available = mocker.patch.object(
            dist, "is_available", return_value=False
        )
        with pytest.raises(ValueError):
            NTXentLoss(gather_distributed=True)
        mock_is_available.assert_called_once()


class TestNTXentLossUnitTest(unittest.TestCase):
    # Old tests in unittest style, please add new tests to TestNTXentLoss using pytest.
    def test_with_values(self):
        for n_samples in [1, 2, 4]:
            for dimension in [1, 2, 16, 64]:
                for temperature in [0.1, 1, 10]:
                    for gather_distributed in [False, True]:
                        out0 = np.random.normal(0, 1, size=(n_samples, dimension))
                        out1 = np.random.normal(0, 1, size=(n_samples, dimension))
                        with self.subTest(
                            msg=(
                                f"out0.shape={out0.shape}, temperature={temperature}, "
                                f"gather_distributed={gather_distributed}"
                            )
                        ):
                            out0 = torch.FloatTensor(out0)
                            out1 = torch.FloatTensor(out1)

                            loss_function = NTXentLoss(
                                temperature=temperature,
                                gather_distributed=gather_distributed,
                            )
                            l1 = float(loss_function(out0, out1))
                            l2 = float(loss_function(out1, out0))
                            l1_manual = self.calc_ntxent_loss_manual(
                                out0, out1, temperature=temperature
                            )
                            l2_manual = self.calc_ntxent_loss_manual(
                                out0, out1, temperature=temperature
                            )
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
                sim = np.inner(z[i], z[j]) / (
                    np.linalg.norm(z[i]) * np.linalg.norm(z[j])
                )
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
        loss /= 2 * N
        return loss

    def test_with_correlated_embedding(self):
        for n_samples in [1, 2, 8, 16]:
            for memory_bank_size in [0, 1, 2, 8, 15, 16, 17]:
                for temperature in [0.1, 1, 7]:
                    for gather_distributed in [False, True]:
                        out0 = np.random.random((n_samples, 1))
                        out1 = np.random.random((n_samples, 1))
                        out0 = np.concatenate([out0, 2 * out0], axis=1)
                        out1 = np.concatenate([out1, 2 * out1], axis=1)
                        out0 = torch.FloatTensor(out0)
                        out1 = torch.FloatTensor(out1)
                        out0.requires_grad = True

                        with self.subTest(
                            msg=(
                                f"n_samples: {n_samples}, memory_bank_size: {memory_bank_size},"
                                f"temperature: {temperature}, gather_distributed: {gather_distributed}"
                            )
                        ):
                            loss_function = NTXentLoss(
                                temperature=temperature,
                                memory_bank_size=memory_bank_size,
                            )
                            if memory_bank_size > 0:
                                for i in range(int(memory_bank_size / n_samples) + 2):
                                    # fill the memory bank over multiple rounds
                                    loss = float(loss_function(out0, out1))
                                expected_loss = -1 * np.log(1 / (memory_bank_size + 1))
                            else:
                                loss = float(loss_function(out0, out1))
                                expected_loss = -1 * np.log(1 / (2 * n_samples - 1))
                            self.assertAlmostEqual(loss, expected_loss, places=5)

    def test_forward_pass(self):
        loss = NTXentLoss(memory_bank_size=0)
        for bsz in range(1, 20):
            batch_1 = torch.randn((bsz, 32))
            batch_2 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0)

    def test_forward_pass_1d(self):
        loss = NTXentLoss(memory_bank_size=0)
        for bsz in range(1, 20):
            batch_1 = torch.randn((bsz, 1))
            batch_2 = torch.randn((bsz, 1))

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0)

    def test_forward_pass_neg_temp(self):
        loss = NTXentLoss(temperature=-1.0, memory_bank_size=0)
        for bsz in range(1, 20):
            batch_1 = torch.randn((bsz, 32))
            batch_2 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0)

    def test_forward_pass_memory_bank(self):
        loss = NTXentLoss(memory_bank_size=64)
        for bsz in range(1, 20):
            batch_1 = torch.randn((bsz, 32))
            batch_2 = torch.randn((bsz, 32))
            l = loss(batch_1, batch_2)

    @unittest.skipUnless(torch.cuda.is_available(), "No cuda")
    def test_forward_pass_memory_bank_cuda(self):
        loss = NTXentLoss(memory_bank_size=64)
        for bsz in range(1, 20):
            batch_1 = torch.randn((bsz, 32)).cuda()
            batch_2 = torch.randn((bsz, 32)).cuda()
            l = loss(batch_1, batch_2)

    @unittest.skipUnless(torch.cuda.is_available(), "No cuda")
    def test_forward_pass_cuda(self):
        loss = NTXentLoss(memory_bank_size=0)
        for bsz in range(1, 20):
            batch_1 = torch.randn((bsz, 32)).cuda()
            batch_2 = torch.randn((bsz, 32)).cuda()

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0)
