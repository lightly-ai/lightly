import pytest
import torch
from pytest_mock import MockerFixture
from torch import distributed as dist
from torch.distributed import nn as dist_nn

from lightly.loss.lejepa_loss import SIGReg


class TestSIGReg:
    def test_backward_pass(self) -> None:
        torch.manual_seed(0)
        loss_fn = SIGReg()
        proj = torch.randn(10, 1024, 16, requires_grad=True)
        loss = loss_fn(proj)
        loss.backward()
        assert proj.grad is not None
        assert proj.grad.shape == proj.shape

    def test_forward(self) -> None:
        torch.manual_seed(0)
        loss_fn = SIGReg(t_max=1.5, num_vectors=32)
        proj = torch.randn(10, 1024, 16)
        loss_fn(proj)

    def test_forward_gather_distributed(self) -> None:
        torch.manual_seed(0)
        loss_fn = SIGReg(gather_distributed=True)
        proj = torch.randn(10, 1024, 16)
        loss = loss_fn(proj)
        assert loss.isfinite()

    def test_forward_gather_distributed_world_size_gt_one(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch("lightly.loss.lejepa_loss.lightly_dist.world_size", return_value=2)
        mock_broadcast = mocker.patch.object(dist, "broadcast")
        mock_all_reduce = mocker.patch.object(dist, "all_reduce")
        mock_dist_nn_all_reduce = mocker.patch.object(
            dist_nn,
            "all_reduce",
            side_effect=lambda tensor, *args, **kwargs: tensor,
        )

        torch.manual_seed(0)
        loss_fn = SIGReg(gather_distributed=True)
        proj = torch.randn(10, 1024, 16)
        loss = loss_fn(proj)

        assert loss.isfinite()
        mock_broadcast.assert_called_once()
        # c10d all_reduce now used only for num_samples_tensor in forward;
        # cos_sum/sin_sum use the autograd-aware variant from torch.distributed.nn.
        assert mock_all_reduce.call_count == 1
        assert mock_dist_nn_all_reduce.call_count == 2

    def test_forward_gather_distributed_matches_non_distributed(
        self, mocker: MockerFixture
    ) -> None:
        """Distributed forward equals non-distributed forward on the global batch.

        Simulates ``world_size=2`` with identical data on both ranks, so the
        global batch is just two copies of the local batch. Verifies the loss
        VALUE under ``gather_distributed=True`` equals the loss on the
        concatenated global batch under ``gather_distributed=False``.
        """
        world_size = 2
        torch.manual_seed(0)
        proj_local = torch.randn(8, 64, 16)
        proj_global = torch.cat([proj_local, proj_local], dim=-2)
        # Use the same projection vectors A in both paths so the two losses
        # are directly comparable.
        A_fixed = torch.randn(proj_local.size(-1), 32)
        A_fixed = A_fixed / A_fixed.norm(p=2, dim=0)

        # Non-distributed truth: SIGReg on the concatenated global batch.
        loss_fn_truth = SIGReg(num_vectors=32, gather_distributed=False)
        mocker.patch.object(
            loss_fn_truth,
            "_generate_unit_vectors",
            new=lambda device, dtype, num_features: A_fixed.to(device, dtype),
        )
        loss_truth = loss_fn_truth(proj_global)

        # Distributed: simulate world_size=2 with identical data on each rank.
        mocker.patch(
            "lightly.loss.lejepa_loss.lightly_dist.world_size",
            return_value=world_size,
        )
        mocker.patch.object(dist, "broadcast")
        mocker.patch.object(
            dist,
            "all_reduce",
            side_effect=lambda tensor, *args, **kwargs: tensor.data.mul_(world_size),
        )
        mocker.patch.object(
            dist_nn,
            "all_reduce",
            side_effect=lambda tensor, *args, **kwargs: tensor * world_size,
        )

        loss_fn_dist = SIGReg(num_vectors=32, gather_distributed=True)
        mocker.patch.object(
            loss_fn_dist,
            "_generate_unit_vectors",
            new=lambda device, dtype, num_features: A_fixed.to(device, dtype),
        )
        loss_dist = loss_fn_dist(proj_local)

        assert torch.allclose(loss_dist, loss_truth, atol=1e-5)

    def test_gather_distributed_gradient_matches_non_distributed(
        self, mocker: MockerFixture
    ) -> None:
        """Distributed gradient equals non-distributed gradient on the global batch.

        Simulates ``world_size=2`` with identical data on both ranks (so the
        global batch is two copies of the local batch). Differentiates a scalar
        parameter ``theta`` and verifies the gradient under
        ``gather_distributed=True`` equals the gradient under
        ``gather_distributed=False`` on the global batch.
        """
        world_size = 2
        torch.manual_seed(0)
        proj_local = torch.randn(8, 64, 16)
        proj_global = torch.cat([proj_local, proj_local], dim=-2)
        A_fixed = torch.randn(proj_local.size(-1), 32)
        A_fixed = A_fixed / A_fixed.norm(p=2, dim=0)

        # Non-distributed truth: gradient on the global batch.
        theta_truth = torch.tensor(2.0, requires_grad=True)
        loss_fn_truth = SIGReg(num_vectors=32, gather_distributed=False)
        mocker.patch.object(
            loss_fn_truth,
            "_generate_unit_vectors",
            new=lambda device, dtype, num_features: A_fixed.to(device, dtype),
        )
        loss_fn_truth(theta_truth * proj_global).backward()
        assert theta_truth.grad is not None

        # Distributed: gradient on the local batch with simulated world_size=2.
        mocker.patch(
            "lightly.loss.lejepa_loss.lightly_dist.world_size",
            return_value=world_size,
        )
        mocker.patch.object(dist, "broadcast")
        mocker.patch.object(
            dist,
            "all_reduce",
            side_effect=lambda tensor, *args, **kwargs: tensor.data.mul_(world_size),
        )
        mocker.patch.object(
            dist_nn,
            "all_reduce",
            side_effect=lambda tensor, *args, **kwargs: tensor * world_size,
        )

        theta_dist = torch.tensor(2.0, requires_grad=True)
        loss_fn_dist = SIGReg(num_vectors=32, gather_distributed=True)
        mocker.patch.object(
            loss_fn_dist,
            "_generate_unit_vectors",
            new=lambda device, dtype, num_features: A_fixed.to(device, dtype),
        )
        loss_fn_dist(theta_dist * proj_local).backward()
        assert theta_dist.grad is not None

        assert torch.allclose(theta_dist.grad, theta_truth.grad, atol=1e-5)

    def test_knots_must_be_greater_than_one(self) -> None:
        with pytest.raises(ValueError):
            SIGReg(knots=1)

    def test_t_max_must_be_greater_than_zero(self) -> None:
        with pytest.raises(ValueError):
            SIGReg(t_max=0.0)

    def test_num_vectors_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            SIGReg(num_vectors=0)
