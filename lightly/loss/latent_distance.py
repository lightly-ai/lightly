"""Distance between predicted and observed frame embeddings.

Every latent world model compares a predicted embedding against an observed
one. They differ only in the norm and in whether the representations are
normalized first, so the comparison lives here rather than in the loss of any
single method.
"""

from __future__ import annotations

from torch import Tensor, nn

DISTANCES = ("l1", "l2")


def latent_distance(
    *,
    predicted: Tensor,
    target: Tensor,
    distance: str = "l2",
    normalize: bool = False,
) -> Tensor:
    """Distance between predicted and target frame embeddings.

    The target is detached, so gradients flow only through the predictor and
    through the encoder path that produced ``predicted``.

    Args:
        predicted: Predicted embeddings of shape ``(B, T, D)``.
        target: Target embeddings with the same shape as ``predicted``.
        distance:
            ``"l1"`` for the mean absolute error, ``"l2"`` for the mean squared
            error. LeWM uses ``"l2"``.
        normalize:
            If True, apply layer normalization to both tensors before the
            comparison. This removes the scale of the representations from the
            loss, which matters for ``"l1"`` because an unnormalized L1 is
            minimized by shrinking the predictions.

    Returns:
        Scalar distance.

    Raises:
        ValueError: If ``distance`` is unknown or the shapes do not match.
    """
    if distance not in DISTANCES:
        raise ValueError(f"distance must be one of {DISTANCES}, got {distance!r}.")
    if predicted.shape != target.shape:
        raise ValueError(
            "predicted and target must have the same shape, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    target = target.detach()
    if normalize:
        normalized_shape = (predicted.size(-1),)
        predicted = nn.functional.layer_norm(predicted, normalized_shape)
        target = nn.functional.layer_norm(target, normalized_shape)
    if distance == "l1":
        return (predicted - target).abs().mean()
    return (predicted - target).square().mean()
