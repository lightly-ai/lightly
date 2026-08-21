"""LeWM loss for latent-space world models."""

from __future__ import annotations

from torch import Tensor, nn

from lightly.loss.latent_distance import latent_distance
from lightly.loss.lejepa_loss import SIGReg


def latent_prediction_loss(
    *, predicted: Tensor, target: Tensor, normalize: bool = False
) -> Tensor:
    """Mean squared error between predicted and target frame embeddings.

    This is :func:`~lightly.loss.latent_distance.latent_distance` with the
    squared error, which is what LeWM uses. The target is detached, so
    gradients flow only through the predictor and through the encoder path that
    produced ``predicted``.

    Args:
        predicted: Predicted embeddings of shape ``(B, T, D)``.
        target: Target embeddings with the same shape as ``predicted``.
        normalize:
            If True, apply layer normalization to both tensors before the
            comparison.

    Returns:
        Scalar prediction loss.
    """
    return latent_distance(
        predicted=predicted, target=target, distance="l2", normalize=normalize
    )


class LeWMLoss(nn.Module):
    """LeWM loss for end-to-end latent world models.

    The loss has two terms:

    - A next-embedding prediction loss between the predicted embeddings and
      the embeddings of the observed next frames.
    - :class:`~lightly.loss.lejepa_loss.SIGReg`, which drives the distribution
      of the frame embeddings toward an isotropic Gaussian.

    SIGReg is what keeps the encoder from collapsing. LeWM therefore needs no
    stop-gradient, no teacher network and no exponential moving average, and
    the encoder can be trained from pixels together with the predictor.

    The total loss is ``prediction + lambda_param * SIGReg(embeddings)``.

    Reference:
        LeWorldModel, 2026, https://arxiv.org/abs/2603.19312

    Attributes:
        lambda_param: Weight of the SIGReg term.
        normalize: Whether the prediction loss normalizes its inputs.
        sigreg: The SIGReg regularizer.

    Examples:
        >>> criterion = LeWMLoss()
        >>>
        >>> # emb has shape (B, T, D), predicted has shape (B, T - 1, D)
        >>> emb = model.encode(frames)
        >>> predicted = model.predict(emb[:, :-1], actions[:, :-1])
        >>>
        >>> loss = criterion(predicted=predicted, target=emb[:, 1:], embeddings=emb)
    """

    def __init__(
        self,
        lambda_param: float = 0.1,
        normalize: bool = False,
        gather_distributed: bool = False,
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        sigreg_num_vectors: int = 1024,
    ) -> None:
        """Initialize the LeWM loss.

        Args:
            lambda_param:
                Weight of the SIGReg term. The LeWM paper uses ``0.1`` and
                reports it as the only hyperparameter that needs tuning.
            normalize:
                If True, layer-normalize the embeddings inside the prediction
                loss.
            gather_distributed:
                If True, aggregate SIGReg statistics across distributed ranks
                so the loss uses the global batch.
            sigreg_knots:
                Number of frequency samples used for the SIGReg integration
                grid.
            sigreg_t_max: Maximum frequency for the SIGReg integration grid.
            sigreg_num_vectors:
                Number of random unit projection vectors used by SIGReg per
                forward pass.
        """
        super().__init__()
        if lambda_param < 0.0:
            raise ValueError("lambda_param must not be negative.")

        self.lambda_param = lambda_param
        self.normalize = normalize
        self.sigreg = SIGReg(
            knots=sigreg_knots,
            t_max=sigreg_t_max,
            num_vectors=sigreg_num_vectors,
            gather_distributed=gather_distributed,
        )

    def forward(
        self, *, predicted: Tensor, target: Tensor, embeddings: Tensor
    ) -> Tensor:
        """Compute the LeWM loss.

        Args:
            predicted: Predicted next-frame embeddings of shape ``(B, T, D)``.
            target:
                Observed next-frame embeddings with the same shape as
                ``predicted``.
            embeddings:
                All frame embeddings of the clip, of shape ``(B, T, D)``, with
                the batch first.

        Returns:
            Scalar loss.
        """
        if embeddings.ndim != 3:
            raise ValueError(
                f"embeddings must have shape (B, T, D), got {tuple(embeddings.shape)}."
            )
        prediction = latent_prediction_loss(
            predicted=predicted, target=target, normalize=self.normalize
        )
        # Every loss in this package takes the batch first. SIGReg draws its
        # samples from dimension -2, so the batch is moved there here rather
        # than asking the caller to do it. Getting this wrong is silent: the
        # statistic would be computed over time instead of over the batch.
        sigreg_input = embeddings.flatten(start_dim=1, end_dim=-2).transpose(0, 1)
        regularization = self.sigreg(sigreg_input)
        total: Tensor = prediction + self.lambda_param * regularization
        return total
