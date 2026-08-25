"""The contracts a backbone satisfies, split by output granularity."""

from __future__ import annotations

from torch import Tensor
from typing_extensions import Protocol, runtime_checkable

__all__ = ["Backbone", "DenseBackbone"]


@runtime_checkable
class Backbone(Protocol):
    """A model that turns images into one pooled vector each.

    Enough for SimCLR, BYOL, MoCo and a kNN or linear probe.

    ``embed`` is deliberately not spelled ``forward_features``: that name is
    timm's, where it returns tokens or a feature map rather than a vector, so a
    raw timm model would satisfy the contract and return the wrong rank.

    Note that ``isinstance`` against this protocol checks member names, never
    signatures. The conformance suite in ``tests/backbones`` is what checks
    behaviour.

    Attributes:
        feature_dim: The width of what ``embed`` returns.
    """

    feature_dim: int

    def embed(self, images: Tensor) -> Tensor:
        """Embeds images into one vector each.

        Args:
            images: Images of shape ``(B, C, H, W)``.

        Returns:
            Pooled features of shape ``(B, feature_dim)``.
        """
        ...


@runtime_checkable
class DenseBackbone(Backbone, Protocol):
    """A backbone that also exposes its spatial features.

    What a dense loss and the segmentation probe read. A ResNet satisfies this;
    the adapter is what normalises NHWC and NCHW to ``(B, N, D)`` plus a grid.
    """

    def feature_map(self, images: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Embeds images into one vector per spatial position.

        Args:
            images: Images of shape ``(B, C, H, W)``.

        Returns:
            Features of shape ``(B, N, D)`` and the ``(height, width)`` grid they
            were flattened from, so that ``N == height * width``.
        """
        ...
