import pytest
import torch
import torch.nn as nn
from packaging.version import Version
from pytest_mock import MockerFixture
from torch import Tensor
from torch import distributed as dist
from torch.nn import Module

from lightly.loss.barlow_twins_loss import BarlowTwinsLoss
from tests.ddp_helpers import NUM_PROCESSES, USE_PYTEST_POOL


class BarlowTwinsLossReference(Module):
    def __init__(
        self,
        projector_dim: int = 8192,
        lambda_param: float = 5e-3,
        gather_distributed: bool = False,
    ):
        super(BarlowTwinsLossReference, self).__init__()
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(projector_dim, affine=False)
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        # code from https://github.com/facebookresearch/barlowtwins/blob/main/main.py

        N = z_a.size(0)

        # empirical cross-correlation matrix
        c = self.bn(z_a).T @ self.bn(z_b)

        # sum the cross-correlation matrix between all gpus
        c.div_(N)
        if self.gather_distributed:
            torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss: Tensor = on_diag + self.lambda_param * off_diag
        return loss


def off_diagonal(x: Tensor) -> Tensor:
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _forward_worker(rank: int, world_size: int, z_a: Tensor, z_b: Tensor) -> Tensor:
    # Pool worker: distributed forward on one rank. See #1982.
    loss: Tensor = BarlowTwinsLoss(gather_distributed=True)(z_a, z_b)
    return loss.detach()


def _gradient_worker(rank: int, world_size: int, z_a: Tensor, z_b: Tensor) -> Tensor:
    # Pool worker: distributed backward on one rank, returns the input gradient.
    z_a = z_a.clone().requires_grad_(True)
    BarlowTwinsLoss(gather_distributed=True)(z_a, z_b).backward()
    assert z_a.grad is not None
    return z_a.grad


class TestBarlowTwinsLoss:
    def test__gather_distributed(self, mocker: MockerFixture) -> None:
        mock_is_available = mocker.patch.object(dist, "is_available", return_value=True)
        BarlowTwinsLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__gather_distributed_dist_not_available(
        self, mocker: MockerFixture
    ) -> None:
        mock_is_available = mocker.patch.object(
            dist, "is_available", return_value=False
        )
        with pytest.raises(ValueError):
            BarlowTwinsLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__loss_matches_reference_loss(self) -> None:
        batch_size = 32
        projector_dim = 8192
        lambda_param = 5e-3
        gather_distributed = False
        loss = BarlowTwinsLoss(
            lambda_param=lambda_param, gather_distributed=gather_distributed
        )
        loss_ref = BarlowTwinsLossReference(
            projector_dim=projector_dim,
            lambda_param=lambda_param,
            gather_distributed=gather_distributed,
        )

        z_a = torch.randn(batch_size, projector_dim)
        z_b = torch.randn(batch_size, projector_dim)

        loss_out = loss(z_a, z_b)
        loss_ref_out = loss_ref(z_a, z_b)

        assert torch.allclose(loss_out, loss_ref_out)

    def test__loss_is_affine_invariant(self) -> None:
        loss = BarlowTwinsLoss()
        x = torch.randn(32, 1024)

        # Loss should be invariant to affine transformations.
        assert torch.allclose(loss(x, x), loss(x, 2 * x + 4))

    @pytest.mark.DDP
    @pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available")
    def test__gather_distributed_forward_matches_non_distributed(self) -> None:
        # With identical data on every rank, the all_reduced cross-correlation
        # matrix equals the non-distributed one, so the forward value matches.
        torch.manual_seed(0)
        z_a = torch.randn(16, 64)
        z_b = torch.randn(16, 64)

        losses = pytest.pool.starmap(  # type: ignore[attr-defined]
            _forward_worker,
            [(rank, NUM_PROCESSES, z_a, z_b) for rank in range(NUM_PROCESSES)],
        )
        loss_truth = BarlowTwinsLoss(gather_distributed=False)(z_a, z_b)

        assert all(torch.allclose(loss, loss_truth, atol=1e-5) for loss in losses)

    @pytest.mark.DDP
    @pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available")
    def test__gather_distributed_gradient_matches_non_distributed(self) -> None:
        # Regression test for the autograd-aware all_reduce fix (#1977): the raw
        # dist.all_reduce left the backward pass unaware of the cross-rank
        # reduction, scaling gradients down by 1/world_size.
        torch.manual_seed(0)
        z_a = torch.randn(16, 64)
        z_b = torch.randn(16, 64)

        grads = pytest.pool.starmap(  # type: ignore[attr-defined]
            _gradient_worker,
            [(rank, NUM_PROCESSES, z_a, z_b) for rank in range(NUM_PROCESSES)],
        )
        z_a_ref = z_a.clone().requires_grad_(True)
        BarlowTwinsLoss(gather_distributed=False)(z_a_ref, z_b).backward()
        assert z_a_ref.grad is not None

        assert all(torch.allclose(grad, z_a_ref.grad, atol=1e-5) for grad in grads)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            torch.float16,
            marks=pytest.mark.skipif(
                Version(torch.__version__.split("+")[0]) < Version("2.3"),
                reason="CPU float16 autocast kernels require PyTorch 2.3 or newer",
            ),
        ),
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize("autocast", [False, True])
def test_barlow_twins_low_precision_reductions(
    dtype: torch.dtype, autocast: bool
) -> None:
    torch.manual_seed(17)
    # The unweighted redundancy sum exceeds float16's finite range, even
    # though the weighted loss and its gradients are representable.
    z_a = torch.randn(16, 1024).to(dtype).float()
    z_b = torch.randn(16, 1024).to(dtype).float()
    reference_a = z_a.clone().requires_grad_()
    reference_b = z_b.clone().requires_grad_()
    reference = BarlowTwinsLoss()(reference_a, reference_b)
    reference.backward()
    actual_a = z_a.to(torch.float32 if autocast else dtype).requires_grad_()
    actual_b = z_b.to(torch.float32 if autocast else dtype).requires_grad_()
    with torch.autocast("cpu", dtype=dtype, enabled=autocast):
        actual = BarlowTwinsLoss()(actual_a, actual_b)
    actual.backward()
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual)
    assert actual_a.grad is not None and actual_b.grad is not None
    assert reference_a.grad is not None and reference_b.grad is not None
    torch.testing.assert_close(actual, reference, rtol=0.005, atol=0.1)
    for actual_grad, reference_grad in (
        (actual_a.grad, reference_a.grad),
        (actual_b.grad, reference_b.grad),
    ):
        assert torch.isfinite(actual_grad).all()
        assert (
            actual_grad.float() - reference_grad
        ).norm() < 0.01 * reference_grad.norm()
        torch.testing.assert_close(
            actual_grad.float(), reference_grad, rtol=0.04, atol=0.004
        )


def test_barlow_twins_preserves_double_precision() -> None:
    torch.manual_seed(18)
    z_a = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
    z_b = torch.randn(8, 16, dtype=torch.float64, requires_grad=True)
    actual = BarlowTwinsLoss()(z_a, z_b)
    expected = BarlowTwinsLossReference(projector_dim=16).double()(z_a, z_b)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected)
