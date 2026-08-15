"""Adapters that make a foreign model satisfy a backbone protocol.

One adapter per model family, named by the user rather than picked by a factory:
a factory taking a model name has to branch on family, and the families do not
agree on what their feature methods return.

    >>> backbone = TorchvisionResNetBackbone(resnet50())
    >>> backbone = TorchvisionResNetBackbone(small_image_stem(resnet18()))
"""

from lightly.backbones.protocols import Backbone, DenseBackbone
from lightly.backbones.torchvision_resnet import (
    TorchvisionResNetBackbone,
    small_image_stem,
)

__all__ = [
    "Backbone",
    "DenseBackbone",
    "TorchvisionResNetBackbone",
    "small_image_stem",
]
