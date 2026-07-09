from __future__ import annotations

from typing import Any

import pytest
import torch
from pytest_mock import MockerFixture
from torch import distributed as dist
from torch.distributions import Normal

from lightly.loss import VISRegLoss
from lightly.loss.lejepa_loss import lejepa_invariance_loss
from lightly.loss.visreg_loss import (
    visreg_center_loss,
    visreg_scale_loss,
    visreg_shape_loss,
)


class TestVisregCenterLoss:
    def test_zero_for_centered_embeddings(self) -> None:
        z = torch.tensor([[1.0, -2.0], [-1.0, 2.0]])
        loss = visreg_center_loss(z)
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_known_value(self) -> None:
        # Constant embeddings of value 2 have a per-dimension mean of 2,
        # so the loss is 2 ** 2 = 4.
        z = torch.full((4, 3), 2.0)
        loss = visreg_center_loss(z)
        assert torch.allclose(loss, torch.tensor(4.0))

    def test_backward(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(8, 16, requires_grad=True)
        loss = visreg_center_loss(z)
        loss.backward()
        assert z.grad is not None
        assert z.grad.shape == z.shape


class TestVisregScaleLoss:
    def test_zero_for_unit_std(self) -> None:
        # Each dimension has values {1, -1} with biased std exactly 1.
        z = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        loss = visreg_scale_loss(z)
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_one_for_collapsed_embeddings(self) -> None:
        z = torch.zeros(8, 4)
        loss = visreg_scale_loss(z)
        assert torch.allclose(loss, torch.tensor(1.0))

    def test_backward(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(8, 16, requires_grad=True)
        loss = visreg_scale_loss(z)
        loss.backward()
        assert z.grad is not None
        assert z.grad.shape == z.shape

    def test_gradient_does_not_vanish_under_collapse(self) -> None:
        # Near-collapsed embeddings must still receive a strong corrective
        # gradient (Figure 2 in the paper). This is the key property that
        # distinguishes VISReg from characteristic-function sketching.
        torch.manual_seed(0)
        z = (1e-8 * torch.randn(32, 16)).requires_grad_()
        loss = visreg_scale_loss(z)
        loss.backward()
        assert z.grad is not None
        assert z.grad.isfinite().all()
        assert z.grad.norm() > 1e-2

    def test_gradient_is_finite_at_exact_collapse(self) -> None:
        # The std backward at exactly zero variance is a 0/0 that PyTorch
        # resolves to zero instead of NaN. Pin this behavior since the loss
        # relies on it to survive fully collapsed embeddings.
        z = torch.full((32, 16), 5.0, requires_grad=True)
        loss = visreg_scale_loss(z)
        loss.backward()
        assert z.grad is not None
        assert z.grad.isfinite().all()


class TestVisregShapeLoss:
    def test_matches_naive_reference(self) -> None:
        # Pin the algorithm against a direct transcription of Algorithm 1
        # from the paper. Reseeding reproduces the same random slices.
        num_slices = 32
        eps = 1e-4
        torch.manual_seed(0)
        z = torch.randn(16, 8)

        torch.manual_seed(42)
        loss = visreg_shape_loss(z, num_slices=num_slices, eps=eps)

        torch.manual_seed(42)
        slices = torch.randn(8, num_slices)
        slices = slices / slices.norm(p=2, dim=0)
        mean = z.mean(dim=0, keepdim=True)
        std = z.std(dim=0, unbiased=False, keepdim=True)
        z_norm = (z - mean) / (std + eps)
        projections_sorted = (z_norm @ slices).sort(dim=0).values
        positions = torch.arange(1, 17, dtype=torch.float32) / 17
        quantiles = Normal(loc=0.0, scale=1.0).icdf(positions).unsqueeze(-1)
        expected = (projections_sorted - quantiles).square().mean()

        assert torch.allclose(loss, expected)

    def test_value_is_scale_invariant(self) -> None:
        # The normalization by the (detached) std makes the loss value
        # independent of the overall scale of the embeddings.
        torch.manual_seed(0)
        z = torch.randn(64, 8)

        torch.manual_seed(42)
        loss = visreg_shape_loss(z, num_slices=16)
        torch.manual_seed(42)
        loss_scaled = visreg_shape_loss(100.0 * z, num_slices=16)

        assert torch.allclose(loss, loss_scaled, atol=1e-4)

    def test_gradient_flows_despite_scale_invariant_value(self) -> None:
        # The stop-gradient on the std means the loss value is scale
        # invariant but the gradient with respect to the embedding scale is
        # not zero. Without the detach the gradient would vanish and the
        # shape loss could not correct a collapsing embedding scale.
        torch.manual_seed(0)
        z = torch.randn(64, 8)
        theta = torch.tensor(1.0, requires_grad=True)

        torch.manual_seed(42)
        loss = visreg_shape_loss(theta * z, num_slices=16)
        loss.backward()

        assert theta.grad is not None
        assert theta.grad.abs() > 1e-4

    def test_gaussian_scores_better_than_exponential(self) -> None:
        torch.manual_seed(0)
        z_gaussian = torch.randn(1024, 8)
        z_exponential = -torch.rand(1024, 8).log()

        torch.manual_seed(42)
        loss_gaussian = visreg_shape_loss(z_gaussian, num_slices=64)
        torch.manual_seed(42)
        loss_exponential = visreg_shape_loss(z_exponential, num_slices=64)

        assert loss_gaussian < loss_exponential

    def test_backward(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(8, 16, requires_grad=True)
        loss = visreg_shape_loss(z, num_slices=16)
        loss.backward()
        assert z.grad is not None
        assert z.grad.shape == z.shape

    def test_num_slices_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            visreg_shape_loss(torch.randn(8, 4), num_slices=0)

    def test_eps_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            visreg_shape_loss(torch.randn(8, 4), num_slices=16, eps=0.0)


class TestVISRegLoss:
    def test_forward(self) -> None:
        torch.manual_seed(0)
        loss_fn = VISRegLoss(num_slices=32)
        local_proj = torch.randn(6, 32, 16)
        global_proj = torch.randn(2, 32, 16)
        loss = loss_fn(local_proj=local_proj, global_proj=global_proj)
        assert loss.isfinite()
        assert loss.ndim == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_forward__cuda(self) -> None:
        torch.manual_seed(0)
        loss_fn = VISRegLoss(num_slices=32)
        local_proj = torch.randn(6, 32, 16).cuda()
        global_proj = torch.randn(2, 32, 16).cuda()
        loss = loss_fn(local_proj=local_proj, global_proj=global_proj)
        assert loss.isfinite()
        assert loss.ndim == 0

    def test_backward_pass(self) -> None:
        torch.manual_seed(0)
        loss_fn = VISRegLoss(num_slices=32)
        local_proj = torch.randn(6, 32, 16, requires_grad=True)
        global_proj = torch.randn(2, 32, 16, requires_grad=True)
        loss = loss_fn(local_proj=local_proj, global_proj=global_proj)
        loss.backward()
        assert local_proj.grad is not None
        assert local_proj.grad.shape == local_proj.shape
        assert global_proj.grad is not None
        assert global_proj.grad.shape == global_proj.shape

    def test_forward_components_weighted_sum(self) -> None:
        # Wiring check with non-default weights so any mixup between the
        # component weights is caught.
        torch.manual_seed(0)
        loss_fn = VISRegLoss(
            lambda_param=0.7,
            num_slices=32,
            lambda_scale=0.5,
            lambda_shape=2.0,
            lambda_center=0.25,
        )
        local_proj = torch.randn(6, 32, 16)
        global_proj = torch.randn(2, 32, 16)

        components = loss_fn.forward_components(
            local_proj=local_proj, global_proj=global_proj
        )
        expected = 0.3 * components.pred + 0.7 * (
            0.5 * components.scale + 2.0 * components.shape + 0.25 * components.center
        )

        assert torch.allclose(components.total, expected)

    def test_forward_matches_components_total(self) -> None:
        torch.manual_seed(0)
        loss_fn = VISRegLoss(num_slices=32)
        local_proj = torch.randn(6, 32, 16)
        global_proj = torch.randn(2, 32, 16)

        torch.manual_seed(42)
        loss = loss_fn(local_proj=local_proj, global_proj=global_proj)
        torch.manual_seed(42)
        components = loss_fn.forward_components(
            local_proj=local_proj, global_proj=global_proj
        )

        assert torch.allclose(loss, components.total)

    def test_lambda_zero_equals_pure_invariance(self) -> None:
        # Eq. 8 includes both global and local views in the prediction term,
        # so the reference invariance loss runs over the concatenated views.
        torch.manual_seed(0)
        local_proj = torch.randn(6, 32, 16)
        global_proj = torch.randn(2, 32, 16)

        loss = VISRegLoss(lambda_param=0.0, num_slices=32)(
            local_proj=local_proj, global_proj=global_proj
        )
        invariance_only = lejepa_invariance_loss(
            local_proj=torch.cat([global_proj, local_proj], dim=0),
            global_proj=global_proj,
        )

        assert torch.allclose(loss, invariance_only)

    @pytest.mark.parametrize("lambda_param", [-0.1, 1.1])
    def test_lambda_must_be_in_unit_interval(self, lambda_param: float) -> None:
        with pytest.raises(ValueError):
            VISRegLoss(lambda_param=lambda_param)

    def test_num_slices_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            VISRegLoss(num_slices=0)

    def test_eps_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            VISRegLoss(eps=0.0)

    @pytest.mark.parametrize(
        "weight_name", ["lambda_scale", "lambda_shape", "lambda_center"]
    )
    def test_component_weights_must_be_non_negative(self, weight_name: str) -> None:
        kwargs: dict[str, Any] = {weight_name: -1.0}
        with pytest.raises(ValueError):
            VISRegLoss(**kwargs)

    def test_batch_size_must_be_greater_than_one(self) -> None:
        loss_fn = VISRegLoss(num_slices=32)
        local_proj = torch.randn(2, 1, 16)
        global_proj = torch.randn(1, 1, 16)
        with pytest.raises(ValueError):
            loss_fn(local_proj=local_proj, global_proj=global_proj)

    @pytest.mark.parametrize(
        ("local_shape", "global_shape"),
        [
            ((32, 16), (2, 32, 16)),
            ((6, 32, 16), (32, 16)),
            ((6, 16, 16), (2, 32, 16)),
            ((6, 32, 8), (2, 32, 16)),
        ],
    )
    def test_validates_projection_shapes(
        self, local_shape: tuple[int, ...], global_shape: tuple[int, ...]
    ) -> None:
        loss_fn = VISRegLoss(num_slices=32)
        local_proj = torch.randn(*local_shape)
        global_proj = torch.randn(*global_shape)
        with pytest.raises(ValueError):
            loss_fn(local_proj=local_proj, global_proj=global_proj)

    def test_default_distributed_path_uses_no_collectives(
        self, mocker: MockerFixture
    ) -> None:
        # Per Section 3.2 of the paper, each device draws its own slices
        # and computes the loss on its local batch. The default path must
        # therefore make no collective calls, unlike the SIGReg pattern
        # (broadcast + all_reduce, see #1920).
        mocker.patch("lightly.loss.visreg_loss.lightly_dist.world_size", return_value=2)
        mock_gather = mocker.patch("lightly.loss.visreg_loss.lightly_dist.gather")
        mock_broadcast = mocker.patch.object(dist, "broadcast")
        mock_all_reduce = mocker.patch.object(dist, "all_reduce")

        torch.manual_seed(0)
        loss_fn = VISRegLoss(num_slices=32)
        local_proj = torch.randn(6, 32, 16)
        global_proj = torch.randn(2, 32, 16)
        loss = loss_fn(local_proj=local_proj, global_proj=global_proj)

        assert loss.isfinite()
        mock_gather.assert_not_called()
        mock_broadcast.assert_not_called()
        mock_all_reduce.assert_not_called()

    def test_gather_distributed_world_size_one_does_not_gather(
        self, mocker: MockerFixture
    ) -> None:
        mock_gather = mocker.patch("lightly.loss.visreg_loss.lightly_dist.gather")

        torch.manual_seed(0)
        loss_fn = VISRegLoss(num_slices=32, gather_distributed=True)
        local_proj = torch.randn(6, 32, 16)
        global_proj = torch.randn(2, 32, 16)
        loss = loss_fn(local_proj=local_proj, global_proj=global_proj)

        assert loss.isfinite()
        mock_gather.assert_not_called()

    def test_gather_distributed_matches_non_distributed(
        self, mocker: MockerFixture
    ) -> None:
        """Gathered forward equals non-distributed forward on the global batch.

        Simulates ``world_size=2`` with identical data on both ranks, so the
        global batch is two copies of the local batch. Reseeding before each
        forward reproduces the same random slices in both paths.
        """
        world_size = 2
        torch.manual_seed(0)
        local_proj = torch.randn(6, 8, 16)
        global_proj = torch.randn(2, 8, 16)
        local_proj_global_batch = torch.cat([local_proj, local_proj], dim=-2)
        global_proj_global_batch = torch.cat([global_proj, global_proj], dim=-2)

        # Non-distributed truth: loss on the concatenated global batch.
        loss_fn_truth = VISRegLoss(num_slices=32, gather_distributed=False)
        torch.manual_seed(42)
        loss_truth = loss_fn_truth(
            local_proj=local_proj_global_batch,
            global_proj=global_proj_global_batch,
        )

        # Distributed: simulate world_size=2 with identical data per rank.
        mocker.patch(
            "lightly.loss.visreg_loss.lightly_dist.world_size",
            return_value=world_size,
        )
        mocker.patch(
            "lightly.loss.visreg_loss.lightly_dist.gather",
            side_effect=lambda tensor: (tensor, tensor),
        )
        loss_fn_dist = VISRegLoss(num_slices=32, gather_distributed=True)
        torch.manual_seed(42)
        loss_dist = loss_fn_dist(local_proj=local_proj, global_proj=global_proj)

        assert torch.allclose(loss_dist, loss_truth, atol=1e-6)

    def test_near_gaussian_projections_give_small_regularization(self) -> None:
        # Standard normal projections are already centered, unit-scale, and
        # Gaussian-shaped, so every regularization component must be small.
        torch.manual_seed(0)
        loss_fn = VISRegLoss(num_slices=64)
        local_proj = torch.randn(2, 2048, 8)
        global_proj = torch.randn(1, 2048, 8)

        components = loss_fn.forward_components(
            local_proj=local_proj, global_proj=global_proj
        )

        assert components.center < 0.01
        assert components.scale < 0.01
        assert components.shape < 0.05

    def test_regularization_includes_global_views(self) -> None:
        # Unlike LeJEPALoss, which regularizes local views only, VISReg
        # regularizes both global and local views (paper Section 3.1 averages
        # over all V views). Changing only the global views must therefore
        # change the regularization components.
        torch.manual_seed(0)
        loss_fn = VISRegLoss(num_slices=32)
        local_proj = torch.randn(2, 32, 16)
        global_proj = torch.randn(2, 32, 16)
        global_proj_shifted = global_proj + 5.0

        torch.manual_seed(1)
        components = loss_fn.forward_components(
            local_proj=local_proj, global_proj=global_proj
        )
        torch.manual_seed(1)
        components_shifted = loss_fn.forward_components(
            local_proj=local_proj, global_proj=global_proj_shifted
        )

        # Shifting the global views changes their mean, so the center loss
        # must react; local-only regularization would leave it unchanged.
        assert not torch.allclose(components.center, components_shifted.center)
