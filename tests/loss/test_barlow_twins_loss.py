import pytest
import torch
import torch.nn as nn
from pytest_mock import MockerFixture
from torch import Tensor
from torch import distributed as dist
from torch.distributed import nn as dist_nn
from torch.nn import Module

from lightly.loss.barlow_twins_loss import BarlowTwinsLoss


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

    def test__gather_distributed_forward_matches_non_distributed(
        self, mocker: MockerFixture
    ) -> None:
        """Distributed forward equals non-distributed forward on the global batch.

        Simulates world_size=2 with identical data on both ranks. The global
        batch is two copies of the local batch, so the cross-correlation matrix
        after the all_reduce equals the non-distributed matrix on the global batch.
        """
        world_size = 2
        torch.manual_seed(0)
        z_a_local = torch.randn(16, 64)
        z_b_local = torch.randn(16, 64)
        z_a_global = torch.cat([z_a_local, z_a_local], dim=0)
        z_b_global = torch.cat([z_b_local, z_b_local], dim=0)

        loss_truth = BarlowTwinsLoss(gather_distributed=False)(z_a_global, z_b_global)

        mocker.patch.object(dist, "is_initialized", return_value=True)
        mocker.patch.object(dist, "get_world_size", return_value=world_size)
        mocker.patch.object(
            dist_nn,
            "all_reduce",
            side_effect=lambda tensor, *args, **kwargs: tensor * world_size,
        )
        loss_dist = BarlowTwinsLoss(gather_distributed=True)(z_a_local, z_b_local)

        assert torch.allclose(loss_dist, loss_truth, atol=1e-5)

    def test__gather_distributed_gradient_matches_non_distributed(
        self, mocker: MockerFixture
    ) -> None:
        """Gradient under gather_distributed=True equals gradient on the global batch.

        Verifies the fix for the autograd bug where raw dist.all_reduce left the
        backward pass unaware of the cross-rank reduction, producing gradients
        scaled down by 1/world_size.
        """
        world_size = 2
        torch.manual_seed(0)
        weight = torch.randn(64, 64)
        z_a_local = torch.randn(16, 64)
        z_b_local = torch.randn(16, 64)
        z_a_global = torch.cat([z_a_local, z_a_local], dim=0)
        z_b_global = torch.cat([z_b_local, z_b_local], dim=0)

        def grad_through_weight(
            loss_fn: BarlowTwinsLoss,
            z_a: torch.Tensor,
            z_b: torch.Tensor,
        ) -> torch.Tensor:
            w = weight.clone().requires_grad_(True)
            loss_fn(z_a @ w, z_b @ w).backward()
            assert w.grad is not None
            return w.grad.clone()

        truth = grad_through_weight(
            BarlowTwinsLoss(gather_distributed=False), z_a_global, z_b_global
        )

        mocker.patch.object(dist, "is_initialized", return_value=True)
        mocker.patch.object(dist, "get_world_size", return_value=world_size)
        mocker.patch.object(
            dist_nn,
            "all_reduce",
            side_effect=lambda tensor, *args, **kwargs: tensor * world_size,
        )
        dist_grad = grad_through_weight(
            BarlowTwinsLoss(gather_distributed=True), z_a_local, z_b_local
        )

        assert torch.allclose(dist_grad, truth, atol=1e-5)
