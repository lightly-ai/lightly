from __future__ import annotations

import pytest
import torch

from lightly.loss import LeWMLoss
from lightly.loss.lewm_loss import latent_prediction_loss


class TestLatentPredictionLoss:
    def test_forward__zero_for_identical_inputs(self) -> None:
        emb = torch.randn(4, 3, 16)
        assert latent_prediction_loss(predicted=emb, target=emb).item() == 0.0

    def test_forward__target_is_detached(self) -> None:
        predicted = torch.randn(4, 3, 16, requires_grad=True)
        target = torch.randn(4, 3, 16, requires_grad=True)
        latent_prediction_loss(predicted=predicted, target=target).backward()
        assert predicted.grad is not None
        assert target.grad is None

    def test_forward__normalize(self) -> None:
        predicted = torch.randn(4, 3, 16)
        # A scaled and shifted copy is identical after layer normalization.
        target = predicted * 3.0 + 1.0
        loss = latent_prediction_loss(
            predicted=predicted, target=target, normalize=True
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_forward__shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            latent_prediction_loss(
                predicted=torch.randn(4, 3, 16), target=torch.randn(4, 2, 16)
            )


class TestLeWMLoss:
    def test_forward(self) -> None:
        torch.manual_seed(0)
        criterion = LeWMLoss(sigreg_num_vectors=32)
        emb = torch.randn(8, 4, 16)
        loss = criterion(
            predicted=torch.randn(8, 3, 16), target=emb[:, 1:], embeddings=emb
        )
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_forward__is_keyword_only(self) -> None:
        """Positional calls must fail loudly rather than swap the arguments."""
        criterion = LeWMLoss(sigreg_num_vectors=32)
        emb = torch.randn(8, 4, 16)
        with pytest.raises(TypeError):
            criterion(torch.randn(8, 3, 16), emb[:, 1:], emb)  # type: ignore[misc]

    def test_forward__invalid_embeddings_shape(self) -> None:
        criterion = LeWMLoss(sigreg_num_vectors=32)
        emb = torch.randn(8, 4, 2, 16)
        with pytest.raises(ValueError, match="embeddings must have shape"):
            criterion(
                predicted=torch.randn(8, 3, 2, 16), target=emb[:, 1:], embeddings=emb
            )

    def test_backward_pass(self) -> None:
        torch.manual_seed(0)
        criterion = LeWMLoss(sigreg_num_vectors=32)
        emb = torch.randn(8, 4, 16, requires_grad=True)
        predicted = torch.randn(8, 3, 16, requires_grad=True)
        criterion(
            predicted=predicted, target=emb[:, 1:].detach(), embeddings=emb
        ).backward()
        assert emb.grad is not None
        assert predicted.grad is not None

    def test_forward__lambda_zero_drops_regularizer(self) -> None:
        torch.manual_seed(0)
        emb = torch.randn(8, 4, 16)
        predicted = torch.randn(8, 3, 16)
        loss = LeWMLoss(lambda_param=0.0, sigreg_num_vectors=32)(
            predicted=predicted, target=emb[:, 1:], embeddings=emb
        )
        expected = latent_prediction_loss(predicted=predicted, target=emb[:, 1:])
        assert loss.item() == pytest.approx(expected.item())

    def test_forward__regularizer_samples_over_the_batch(self) -> None:
        """SIGReg must read the batch axis, not the time axis.

        The loss moves the batch to where SIGReg wants it. A caller cannot see
        the difference, so a mix-up here would be silent: the statistic would
        be computed over the frames of a clip instead of over the batch.
        """
        torch.manual_seed(0)
        criterion = LeWMLoss(lambda_param=1.0, sigreg_num_vectors=32)
        predicted = torch.zeros(8, 3, 16)

        # Collapsed over the batch: every sample is the same frame sequence.
        collapsed_batch = torch.randn(1, 4, 16).expand(8, 4, 16)
        # Collapsed over time: every frame of a sample is the same vector.
        collapsed_time = torch.randn(8, 1, 16).expand(8, 4, 16)

        torch.manual_seed(1)
        batch_loss = criterion(
            predicted=predicted,
            target=collapsed_batch[:, 1:],
            embeddings=collapsed_batch,
        )
        torch.manual_seed(1)
        time_loss = criterion(
            predicted=predicted, target=collapsed_time[:, 1:], embeddings=collapsed_time
        )
        # Only the batch-collapsed input is degenerate for a batch statistic.
        assert batch_loss.item() > time_loss.item()

    def test_init__negative_lambda(self) -> None:
        with pytest.raises(ValueError, match="lambda_param must not be negative"):
            LeWMLoss(lambda_param=-1.0)
