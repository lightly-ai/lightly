import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture
from torch import Tensor
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

    @pytest.mark.parametrize("n_samples", [1, 2, 4])
    @pytest.mark.parametrize("dimension", [1, 2, 16, 64])
    @pytest.mark.parametrize("temperature", [0.1, 1, 10])
    @pytest.mark.parametrize("gather_distributed", [False, True])
    def test_with_values(
        self,
        n_samples: int,
        dimension: int,
        temperature: float,
        gather_distributed: bool,
    ) -> None:
        out0 = torch.tensor(
            np.random.normal(0, 1, size=(n_samples, dimension)), dtype=torch.float32
        )
        out1 = torch.tensor(
            np.random.normal(0, 1, size=(n_samples, dimension)), dtype=torch.float32
        )

        loss_function = NTXentLoss(
            temperature=temperature,
            gather_distributed=gather_distributed,
        )
        l1 = float(loss_function(out0, out1))
        l2 = float(loss_function(out1, out0))
        l1_manual = _calc_ntxent_loss_manual(out0, out1, temperature=temperature)
        l2_manual = _calc_ntxent_loss_manual(out0, out1, temperature=temperature)
        assert l1 == pytest.approx(l2, abs=1e-5)
        assert l1 == pytest.approx(l1_manual, abs=1e-5)
        assert l2 == pytest.approx(l2_manual, abs=1e-5)

    @pytest.mark.parametrize("n_samples", [1, 2, 8, 16])
    @pytest.mark.parametrize("memory_bank_size", [0, 1, 2, 8, 15, 16, 17])
    @pytest.mark.parametrize("temperature", [0.1, 1, 7])
    @pytest.mark.parametrize("gather_distributed", [False, True])
    def test_with_correlated_embedding(
        self,
        n_samples: int,
        memory_bank_size: int,
        temperature: float,
        gather_distributed: bool,
    ) -> None:
        out0_np = np.random.random((n_samples, 1))
        out1_np = np.random.random((n_samples, 1))
        out0_np = np.concatenate([out0_np, 2 * out0_np], axis=1)
        out1_np = np.concatenate([out1_np, 2 * out1_np], axis=1)
        out0: Tensor = torch.tensor(out0_np, dtype=torch.float32)
        out1: Tensor = torch.tensor(out1_np, dtype=torch.float32)
        out0.requires_grad = True
        loss_function = NTXentLoss(
            temperature=temperature,
            memory_bank_size=(memory_bank_size, 2) if memory_bank_size > 0 else 0,
            gather_distributed=gather_distributed,
        )
        if memory_bank_size > 0:
            for _ in range(int(memory_bank_size / n_samples) + 2):
                # fill the memory bank over multiple rounds
                loss = float(loss_function(out0, out1))
            expected_loss = -1 * np.log(1 / (memory_bank_size + 1))
        else:
            loss = float(loss_function(out0, out1))
            expected_loss = -1 * np.log(1 / (2 * n_samples - 1))
        assert loss == pytest.approx(expected_loss)

    def test_forward_pass(self) -> None:
        loss = NTXentLoss(memory_bank_size=0)
        for bsz in range(1, 20):
            batch_1 = torch.randn((bsz, 32))
            batch_2 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            assert (l1 - l2).pow(2).item() == pytest.approx(0.0)

    def test_forward_pass_1d(self) -> None:
        loss = NTXentLoss(memory_bank_size=0)
        for bsz in range(1, 20):
            batch_1 = torch.randn((bsz, 1))
            batch_2 = torch.randn((bsz, 1))

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            assert (l1 - l2).pow(2).item() == pytest.approx(0.0)

    def test_forward_pass_neg_temp(self) -> None:
        loss = NTXentLoss(temperature=-1.0, memory_bank_size=0)
        for bsz in range(1, 20):
            batch_1 = torch.randn((bsz, 32))
            batch_2 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            assert (l1 - l2).pow(2).item() == pytest.approx(0.0)

    def test_forward_pass_memory_bank(self) -> None:
        loss = NTXentLoss(memory_bank_size=(64, 32))
        for bsz in range(1, 20):
            batch_1 = torch.randn((bsz, 32))
            batch_2 = torch.randn((bsz, 32))
            l = loss(batch_1, batch_2)

    @pytest.mark.parametrize(
        "normalize_eps", [0.0, -1e-12, float("nan"), float("inf"), float("-inf")]
    )
    def test_normalize_eps__invalid(self, normalize_eps: float) -> None:
        with pytest.raises(ValueError, match="Illegal normalize_eps"):
            NTXentLoss(normalize_eps=normalize_eps)

    def test_normalize_eps(self) -> None:
        # The norm of the embeddings is far below normalize_eps, so they are divided
        # by normalize_eps instead of being normalized to unit length. All
        # similarities are then close to zero and the loss is close to log(2N - 1).
        n_samples = 8
        out0 = 1e-3 * torch.randn(n_samples, 16)
        out1 = out0.clone()
        uniform_loss = np.log(2 * n_samples - 1)

        loss_default = NTXentLoss()(out0, out1)
        loss_large_eps = NTXentLoss(normalize_eps=1.0)(out0, out1)

        assert float(loss_default) < uniform_loss - 0.5
        assert float(loss_large_eps) == pytest.approx(uniform_loss, abs=1e-3)

    def test_normalize_eps__float16_zero_embedding(self) -> None:
        # 1e-12 rounds to 0 in float16, so with the default normalize_eps an all-zero
        # embedding is divided by zero and the loss is NaN.
        ones = torch.ones(2, 2, dtype=torch.float16)
        try:
            ones @ ones
        except RuntimeError:
            pytest.skip("float16 matmul is not supported on CPU by this torch version.")
        out0 = torch.randn(8, 16).half()
        out1 = torch.randn(8, 16).half()
        out0[0] = 0

        loss = NTXentLoss(normalize_eps=1e-6)(out0, out1)

        assert torch.isfinite(loss)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_forward_pass_memory_bank_cuda(self) -> None:
        loss = NTXentLoss(memory_bank_size=(64, 32))
        for bsz in range(1, 20):
            batch_1 = torch.randn((bsz, 32)).cuda()
            batch_2 = torch.randn((bsz, 32)).cuda()
            l = loss(batch_1, batch_2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_forward_pass_cuda(self) -> None:
        loss = NTXentLoss(memory_bank_size=0)
        for bsz in range(1, 20):
            batch_1 = torch.randn((bsz, 32)).cuda()
            batch_2 = torch.randn((bsz, 32)).cuda()

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            assert (l1 - l2).pow(2).item() == pytest.approx(0.0)


def _calc_ntxent_loss_manual(
    out0_torch: Tensor, out1_torch: Tensor, temperature: float
) -> float:
    # using the pseudocode directly from https://arxiv.org/pdf/2002.05709.pdf , Algorithm 1

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
