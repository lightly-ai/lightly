import numpy as np
import pytest
import torch
from torch import Tensor

from lightly.loss import DirectCLRLoss


class TestDirectCLRLoss:
    temperature = 0.5

    def test_set_parent_temperature(self) -> None:
        loss_function = DirectCLRLoss(loss_dim=32, temperature=self.temperature)
        assert (
            loss_function.temperature == self.temperature
        ), f"Expected temperature to be {self.temperature}, but got {loss_function.temperature}"

    @pytest.mark.parametrize("n_samples", [1, 2, 4])
    @pytest.mark.parametrize("dimension", [1, 2, 8])
    @pytest.mark.parametrize("loss_dim", [1, 2, 4])
    def test_with_values(
        self,
        n_samples: int,
        dimension: int,
        loss_dim: int,
    ) -> None:
        out0 = torch.randn((n_samples, dimension))
        out1 = torch.randn((n_samples, dimension))

        loss_function = DirectCLRLoss(loss_dim=loss_dim, temperature=self.temperature)
        l1 = float(loss_function(out0, out1))
        l2 = float(loss_function(out1, out0))
        l1_manual = _calc_directclr_loss_manual(
            out0, out1, loss_dim=loss_dim, temperature=self.temperature
        )
        l2_manual = _calc_directclr_loss_manual(
            out0, out1, loss_dim=loss_dim, temperature=self.temperature
        )
        assert l1 == pytest.approx(l2, abs=1e-5)
        assert l1 == pytest.approx(l1_manual, abs=1e-5)
        assert l2 == pytest.approx(l2_manual, abs=1e-5)

    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_forward_pass(self, batch_size: int) -> None:
        loss = DirectCLRLoss(loss_dim=32)
        batch_1 = torch.randn((batch_size, 64))
        batch_2 = torch.randn((batch_size, 64))

        # symmetry
        l1 = loss(batch_1, batch_2)
        l2 = loss(batch_2, batch_1)
        assert (l1 - l2).pow(2).item() == pytest.approx(0.0)

    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_forward_pass_1d(self, batch_size: int) -> None:
        loss = DirectCLRLoss(loss_dim=32)
        batch_1 = torch.randn((batch_size, 1))
        batch_2 = torch.randn((batch_size, 1))

        # symmetry
        l1 = loss(batch_1, batch_2)
        l2 = loss(batch_2, batch_1)
        assert (l1 - l2).pow(2).item() == pytest.approx(0.0)


def _calc_directclr_loss_manual(
    out0_torch: Tensor, out1_torch: Tensor, loss_dim: int, temperature: float
) -> float:
    # Identical to the _calc_ntxent_loss_manual calculations from test_ntx_ent_loss.py
    # using the pseudocode directly from https://arxiv.org/pdf/2002.05709.pdf , Algorithm 1

    out0_torch = out0_torch.flatten(start_dim=1)[:, :loss_dim]
    out1_torch = out1_torch.flatten(start_dim=1)[:, :loss_dim]

    out0 = np.array(out0_torch)
    out1 = np.array(out1_torch)

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

    loss = 0.0
    for k in range(N):
        loss += l_i_j[k, k + N] + l_i_j[k + N, k]
    loss /= 2.0 * float(N)
    return loss
