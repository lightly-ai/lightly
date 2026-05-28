import math
from typing import Dict

import pytest
import torch

from lightly.loss.regularizer import DSERegularizer


class TestDSERegularizer:
    def test_forward_pass(self) -> None:
        torch.manual_seed(0)
        regularizer = DSERegularizer(
            local_clusters=2,
            global_clusters=4,
            max_kmeans_iters=3,
        )
        representations = torch.randn(4, 8, 6)

        loss = regularizer(representations)

        assert loss.ndim == 0
        assert loss.isfinite()

    def test_backward_pass(self) -> None:
        torch.manual_seed(0)
        regularizer = DSERegularizer(
            local_clusters=2,
            global_clusters=4,
            max_kmeans_iters=3,
        )
        representations = torch.randn(4, 8, 6, requires_grad=True)

        loss = regularizer(representations)
        loss.backward()

        assert representations.grad is not None
        assert representations.grad.shape == representations.shape

    def test_convolutional_features_can_be_flattened(self) -> None:
        torch.manual_seed(0)
        regularizer = DSERegularizer(
            local_clusters=2,
            global_clusters=4,
            max_kmeans_iters=3,
        )
        features = torch.randn(4, 6, 3, 3)
        dense_features = features.flatten(2).transpose(1, 2)

        loss = regularizer(dense_features)

        assert loss.ndim == 0
        assert loss.isfinite()

    def test_weight_zero_returns_zero_loss(self) -> None:
        regularizer = DSERegularizer(weight=0.0)
        representations = torch.randn(4, 8, 6)

        loss = regularizer(representations)

        assert loss.item() == 0.0

    def test_weight_scales_loss(self) -> None:
        representations = torch.randn(4, 8, 6)
        regularizer = DSERegularizer(
            weight=1.0,
            local_clusters=2,
            global_clusters=4,
            max_kmeans_iters=3,
        )
        weighted_regularizer = DSERegularizer(
            weight=2.0,
            local_clusters=2,
            global_clusters=4,
            max_kmeans_iters=3,
        )

        torch.manual_seed(0)
        loss = regularizer(representations)
        torch.manual_seed(0)
        weighted_loss = weighted_regularizer(representations)

        assert torch.allclose(weighted_loss, 2 * loss)

    def test_invalid_input_shape_raises(self) -> None:
        regularizer = DSERegularizer()

        with pytest.raises(ValueError):
            regularizer(torch.randn(4, 6))

    def test_empty_input_raises(self) -> None:
        regularizer = DSERegularizer()

        with pytest.raises(ValueError):
            regularizer(torch.randn(4, 0, 6))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"weight": -1.0},
            {"local_clusters": 0},
            {"global_clusters": 0},
            {"max_kmeans_iters": 0},
            {"tol": 0.0},
        ],
    )
    def test_invalid_init_args_raise(self, kwargs: Dict[str, object]) -> None:
        with pytest.raises(ValueError):
            DSERegularizer(**kwargs)  # type: ignore[arg-type]

    def test_effective_dimensionality_matches_reference(self) -> None:
        # Verifies _effective_dimensionality against the formula from
        # SSL-Degradation/DSE_regularizer.py (estimator, 'effective_rank' branch).
        torch.manual_seed(0)
        x = torch.randn(16, 8)
        regularizer = DSERegularizer()

        result = regularizer._effective_dimensionality(x)

        sv = torch.linalg.svdvals(x)
        p = sv / (sv.sum() + 1e-12) + 1e-12
        expected = torch.exp(-torch.sum(p * torch.log(p))) / min(x.shape)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_centered_singular_sum_matches_reference(self) -> None:
        # Verifies _centered_singular_sum against the formula from
        # SSL-Degradation/DSE_regularizer.py (estimator, 'centered_singular_sum' branch).
        torch.manual_seed(0)
        x = torch.randn(16, 8)
        regularizer = DSERegularizer()

        result = regularizer._centered_singular_sum(x)

        xc = x - x.mean(dim=0, keepdim=True)
        sv = torch.linalg.svdvals(xc)
        n, d = x.shape
        expected = sv.sum() / math.sqrt(max(n - 1, d))
        assert torch.allclose(result, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_forward_pass_cuda(self) -> None:
        torch.manual_seed(0)
        regularizer = DSERegularizer(
            local_clusters=2,
            global_clusters=4,
            max_kmeans_iters=3,
        ).cuda()
        representations = torch.randn(4, 8, 6, device="cuda")

        loss = regularizer(representations)

        assert loss.ndim == 0
        assert loss.isfinite()
