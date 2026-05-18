import pytest
import torch
from pytest_mock import MockerFixture
from torch import distributed as dist
from torch.distributed import nn as dist_nn

from lightly.loss import LeJEPALoss, SIGReg
from lightly.loss.lejepa_loss import lejepa_invariance_loss


class TestSIGReg:
    # TODO: the distributed-primitive mocks in the gather_distributed tests
    # below approximate (but don't fully match) the real c10d/torch.distributed.nn APIs.
    # Replace them with real Gloo-backed DDP tests once that infrastructure
    # is available (see PR #1923 review).

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
        # TODO: replace with Gloo DDP testing (see class-level note).
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
        # TODO: replace with Gloo DDP testing (see class-level note).
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
        # TODO: replace with Gloo DDP testing (see class-level note).
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


class TestLeJEPAInvarianceLoss:
    def test_forward(self) -> None:
        torch.manual_seed(0)
        proj = torch.randn(8, 32, 128)
        loss = lejepa_invariance_loss(proj)
        assert loss.isfinite()
        assert loss.ndim == 0

    def test_backward(self) -> None:
        torch.manual_seed(0)
        proj = torch.randn(8, 32, 128, requires_grad=True)
        loss = lejepa_invariance_loss(proj)
        loss.backward()
        assert proj.grad is not None
        assert proj.grad.shape == proj.shape


class TestLeJEPALoss:
    def test_backward_pass(self) -> None:
        torch.manual_seed(0)
        loss_fn = LeJEPALoss()
        proj = torch.randn(8, 32, 128, requires_grad=True)
        loss = loss_fn(proj)
        loss.backward()
        assert proj.grad is not None
        assert proj.grad.shape == proj.shape

    def test_forward(self) -> None:
        torch.manual_seed(0)
        loss_fn = LeJEPALoss()
        proj = torch.randn(8, 32, 128)
        loss = loss_fn(proj)
        assert loss.isfinite()

    def test_forward_gather_distributed_world_size_gt_one(
        self, mocker: MockerFixture
    ) -> None:
        # TODO: replace with Gloo DDP testing (see class-level note on TestSIGReg).
        mocker.patch("lightly.loss.lejepa_loss.lightly_dist.world_size", return_value=2)
        mock_broadcast = mocker.patch.object(dist, "broadcast")
        mock_all_reduce = mocker.patch.object(dist, "all_reduce")
        mock_dist_nn_all_reduce = mocker.patch.object(
            dist_nn,
            "all_reduce",
            side_effect=lambda tensor, *args, **kwargs: tensor,
        )

        torch.manual_seed(0)
        loss_fn = LeJEPALoss(gather_distributed=True)
        proj = torch.randn(8, 32, 128)
        loss = loss_fn(proj)

        assert loss.isfinite()
        mock_broadcast.assert_called_once()
        # c10d all_reduce is used only for num_samples_tensor in SIGReg.forward;
        # cos_sum/sin_sum use the autograd-aware variant from torch.distributed.nn.
        assert mock_all_reduce.call_count == 1
        assert mock_dist_nn_all_reduce.call_count == 2

    @pytest.mark.parametrize("lambda_param", [-0.1, 1.1])
    def test_lambda_must_be_in_unit_interval(self, lambda_param: float) -> None:
        with pytest.raises(ValueError):
            LeJEPALoss(lambda_param=lambda_param)

    def test_lambda_zero_equals_pure_invariance(self) -> None:
        # Wiring check: at lambda=0 the SIGReg term is zeroed out, so the loss
        # must equal the standalone invariance loss on the same projections.
        torch.manual_seed(0)
        proj = torch.randn(8, 32, 128)

        lejepa_loss = LeJEPALoss(lambda_param=0.0)(proj)
        invariance_only = lejepa_invariance_loss(proj)

        assert torch.allclose(lejepa_loss, invariance_only)

    def test_lambda_one_equals_pure_sigreg(self) -> None:
        # Wiring check: at lambda=1 the invariance term is zeroed out, so the
        # loss must equal a standalone SIGReg call under the same RNG state.
        lejepa_fn = LeJEPALoss(lambda_param=1.0)
        sigreg_fn = SIGReg()
        torch.manual_seed(0)
        proj = torch.randn(8, 32, 128)

        torch.manual_seed(42)
        lejepa_loss = lejepa_fn(proj)

        torch.manual_seed(42)
        sigreg_loss = sigreg_fn(proj)

        assert torch.allclose(lejepa_loss, sigreg_loss)
