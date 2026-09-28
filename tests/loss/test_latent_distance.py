from __future__ import annotations

import pytest
import torch

from lightly.loss.latent_distance import latent_distance


class TestLatentDistance:
    @pytest.mark.parametrize("distance", ["l1", "l2"])
    def test_forward__zero_for_identical_inputs(self, distance: str) -> None:
        emb = torch.randn(4, 3, 16)
        loss = latent_distance(predicted=emb, target=emb, distance=distance)
        assert loss.item() == 0.0

    def test_forward__l1_is_the_mean_absolute_error(self) -> None:
        predicted = torch.zeros(2, 3, 4)
        target = torch.full((2, 3, 4), 2.0)
        loss = latent_distance(predicted=predicted, target=target, distance="l1")
        assert loss.item() == pytest.approx(2.0)

    def test_forward__l2_is_the_mean_squared_error(self) -> None:
        predicted = torch.zeros(2, 3, 4)
        target = torch.full((2, 3, 4), 2.0)
        loss = latent_distance(predicted=predicted, target=target, distance="l2")
        assert loss.item() == pytest.approx(4.0)

    @pytest.mark.parametrize("distance", ["l1", "l2"])
    def test_forward__target_is_detached(self, distance: str) -> None:
        predicted = torch.randn(4, 3, 16, requires_grad=True)
        target = torch.randn(4, 3, 16, requires_grad=True)
        latent_distance(
            predicted=predicted, target=target, distance=distance
        ).backward()
        assert predicted.grad is not None
        assert target.grad is None

    @pytest.mark.parametrize("distance", ["l1", "l2"])
    def test_forward__normalize_removes_scale_and_shift(self, distance: str) -> None:
        predicted = torch.randn(4, 3, 16)
        target = predicted * 3.0 + 1.0
        loss = latent_distance(
            predicted=predicted, target=target, distance=distance, normalize=True
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_forward__unknown_distance(self) -> None:
        with pytest.raises(ValueError, match="distance must be one of"):
            latent_distance(
                predicted=torch.randn(4, 3, 16),
                target=torch.randn(4, 3, 16),
                distance="cosine",
            )

    def test_forward__shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            latent_distance(
                predicted=torch.randn(4, 3, 16), target=torch.randn(4, 2, 16)
            )
