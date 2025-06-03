import pytest
import torch
from pytest_mock import MockerFixture
from torch import distributed as dist

from lightly.loss.dcl_loss import DCLLoss, DCLWLoss, negative_mises_fisher_weights


class TestDCLLoss:
    def test__gather_distributed(self, mocker: MockerFixture) -> None:
        mock_is_available = mocker.patch.object(dist, "is_available", return_value=True)
        DCLLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__gather_distributed_dist_not_available(
        self, mocker: MockerFixture
    ) -> None:
        mock_is_available = mocker.patch.object(
            dist, "is_available", return_value=False
        )
        with pytest.raises(ValueError):
            DCLLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    @pytest.mark.parametrize("sigma", [0.0000001, 0.5, 10000])
    def test_negative_mises_fisher_weights(self, sigma: float, seed: int = 0) -> None:
        torch.manual_seed(seed)
        out0 = torch.rand((3, 5))
        out1 = torch.rand((3, 5))
        negative_mises_fisher_weights(out0, out1, sigma)

    @pytest.mark.parametrize("batch_size", [2, 3])
    @pytest.mark.parametrize("dim", [1, 3])
    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("gather_distributed", [False, True])
    def test_dclloss_forward(
        self,
        batch_size: int,
        dim: int,
        temperature: float,
        gather_distributed: bool,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed=seed)
        out0 = torch.rand((batch_size, dim))
        out1 = torch.rand((batch_size, dim))
        criterion = DCLLoss(
            temperature=temperature,
            gather_distributed=gather_distributed,
            weight_fn=negative_mises_fisher_weights,
        )
        loss0 = criterion(out0, out1)
        loss1 = criterion(out1, out0)
        assert loss0 > 0
        assert loss0 == pytest.approx(loss1)

    @pytest.mark.parametrize("batch_size", [2, 3])
    @pytest.mark.parametrize("dim", [1, 3])
    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("gather_distributed", [False, True])
    def test_dclloss_forward__no_weight_fn(
        self,
        batch_size: int,
        dim: int,
        temperature: float,
        gather_distributed: bool,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed=seed)
        out0 = torch.rand((batch_size, dim))
        out1 = torch.rand((batch_size, dim))
        criterion = DCLLoss(
            temperature=temperature,
            gather_distributed=gather_distributed,
            weight_fn=None,
        )
        loss0 = criterion(out0, out1)
        loss1 = criterion(out1, out0)
        assert loss0 > 0
        assert loss0 == pytest.approx(loss1)

    def test_dclloss_backprop(self, seed: int = 0) -> None:
        torch.manual_seed(seed=seed)
        out0 = torch.rand(3, 5)
        out1 = torch.rand(3, 5)
        layer = torch.nn.Linear(5, 5)
        out0 = layer(out0)
        out1 = layer(out1)
        criterion = DCLLoss()
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        loss = criterion(out0, out1)
        loss.backward()
        optimizer.step()

    def test_dclwloss_forward(self, seed: int = 0) -> None:
        torch.manual_seed(seed=seed)
        out0 = torch.rand(3, 5)
        out1 = torch.rand(3, 5)
        criterion = DCLWLoss()
        loss0 = criterion(out0, out1)
        loss1 = criterion(out1, out0)
        assert loss0 > 0
        assert loss0 == pytest.approx(loss1)

    @pytest.mark.parametrize(
        "batch_size, dim, temperature",
        [
            (3, 5, 0.1),
            (4, 7, 0.5),
        ],
    )
    def test_dclloss_matches_reference(
        self, batch_size: int, dim: int, temperature: float, seed: int = 0
    ) -> None:
        """
        Ensure patched DCLLoss matches loop-based reference for multiple configs.
        """
        torch.manual_seed(seed)
        out0 = torch.randn(batch_size, dim)
        out1 = torch.randn(batch_size, dim)

        loss_fn = DCLLoss(temperature=temperature)
        patched = loss_fn(out0, out1)

        ref = dcl_reference(out0, out1, temperature)
        assert torch.allclose(
            patched, ref
        ), f"patch={patched.item():.6f} vs ref={ref.item():.6f}"


def dcl_reference(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float
) -> torch.Tensor:
    """
    Loop-based reference:
      L = avg_i[ -sim_pos(i) + log(sum_j exp(sim_neg(i,j))) ]
    """
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # 2N × D

    losses = []
    for i in range(2 * N):
        # find positive index
        pos_idx = i + N if i < N else i - N
        sim_pos = torch.dot(z[i], z[pos_idx]) / temperature

        # negatives are all except i and pos_idx
        mask = torch.ones(2 * N, dtype=torch.bool)
        mask[i] = False
        mask[pos_idx] = False
        sims = torch.matmul(z[i], z[mask].T) / temperature  # shape (2N-2,)
        all_dims = tuple(range(sims.ndim))
        neg_term = torch.logsumexp(sims, dim=all_dims)
        losses.append(-sim_pos + neg_term)

    return torch.stack(losses).mean()
