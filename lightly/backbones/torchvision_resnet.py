"""Adapter for the torchvision ResNet family."""

from __future__ import annotations

from torch import Tensor
from torch.nn import Conv2d, Identity, Module
from torchvision.models.resnet import ResNet

__all__ = ["TorchvisionResNetBackbone", "small_image_stem"]


class TorchvisionResNetBackbone(Module):
    """Adapts a torchvision ResNet to ``Backbone`` and ``DenseBackbone``.

    The classification head is replaced by an identity, so the wrapped model
    returns pooled features. The width is read off the head before it goes.

    Example:
        >>> backbone = TorchvisionResNetBackbone(resnet50())
        >>> backbone.feature_dim
        2048
        >>> backbone.embed(images).shape
        torch.Size([8, 2048])

    Args:
        model:
            A torchvision ResNet, mutated in place. Wrap the stem first if the
            images are small, see :func:`small_image_stem`.

    Raises:
        ValueError: If the model has no classification head to read the width
            from, because it is not a ResNet or its head was already replaced.
    """

    def __init__(self, model: ResNet) -> None:
        super().__init__()
        head = getattr(model, "fc", None)
        in_features = getattr(head, "in_features", None)
        if in_features is None:
            raise ValueError(
                f"cannot read feature_dim: {type(model).__name__}.fc is "
                f"{type(head).__name__}, not a Linear. Pass a torchvision ResNet "
                "with its classification head still attached."
            )
        self.feature_dim: int = in_features
        model.fc = Identity()
        self.model = model

    def embed(self, images: Tensor) -> Tensor:
        """Embeds images into one vector each.

        Goes through ``__call__`` rather than around it, so a forward hook
        registered on this module sees the call.

        Args:
            images: Images of shape ``(B, C, H, W)``.

        Returns:
            Pooled features of shape ``(B, feature_dim)``.
        """
        features: Tensor = self(images)
        return features

    def feature_map(self, images: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Embeds images into one vector per spatial position.

        Args:
            images: Images of shape ``(B, C, H, W)``.

        Returns:
            Features of shape ``(B, N, feature_dim)`` and the ``(height, width)``
            grid they were flattened from.
        """
        model = self.model
        x = model.maxpool(model.relu(model.bn1(model.conv1(images))))
        x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
        height, width = x.shape[-2:]
        return x.flatten(2).transpose(1, 2), (height, width)

    def forward(self, images: Tensor) -> Tensor:
        """Same as :meth:`embed`, so the adapter drops into ``nn.Sequential``.

        Args:
            images: Images of shape ``(B, C, H, W)``.

        Returns:
            Pooled features of shape ``(B, feature_dim)``.
        """
        features: Tensor = self.model(images)
        return features


def small_image_stem(model: ResNet) -> ResNet:
    """Swaps a ResNet's stem for the one used on small images.

    A 3x3 stride-1 convolution and no max pooling, which keeps a 32-pixel image
    from being reduced to a 1x1 grid before the first block. This is what
    ``ResNetGenerator("resnet-18")`` did for the CIFAR-10 benchmark.

    Args:
        model: A torchvision ResNet, mutated in place.

    Returns:
        The same model, so it composes with the adapter on one line.
    """
    conv1 = model.conv1
    model.conv1 = Conv2d(
        in_channels=conv1.in_channels,
        out_channels=conv1.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = Identity()
    return model
