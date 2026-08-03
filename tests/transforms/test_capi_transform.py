from __future__ import annotations

import torch
from PIL import Image

from lightly.transforms import CAPITransform
from lightly.transforms.utils import IMAGENET_NORMALIZE


def test_view_on_pil_image() -> None:
    transform = CAPITransform(input_size=32)
    output = transform(Image.new("RGB", (100, 100)))
    assert len(output) == 1
    assert output[0].shape == (3, 32, 32)


def test_default_input_size() -> None:
    transform = CAPITransform()
    output = transform(Image.new("RGB", (256, 256)))
    assert output[0].shape == (3, 224, 224)


def test_normalize_none() -> None:
    transform = CAPITransform(input_size=32, normalize=None)
    output = transform(Image.new("RGB", (100, 100), color=(255, 255, 255)))[0]
    # Without normalization a white image stays at 1.0 after ToTensor.
    assert torch.allclose(output, torch.ones_like(output), atol=1e-6)


def test_normalize_default() -> None:
    transform = CAPITransform(input_size=32, normalize=IMAGENET_NORMALIZE)
    output = transform(Image.new("RGB", (100, 100), color=(255, 255, 255)))[0]
    # A white image is 1.0 after ToTensor; ImageNet normalization maps each
    # channel c to (1 - mean[c]) / std[c].
    mean = torch.tensor(IMAGENET_NORMALIZE["mean"]).view(3, 1, 1)
    std = torch.tensor(IMAGENET_NORMALIZE["std"]).view(3, 1, 1)
    expected = (1.0 - mean) / std
    assert torch.allclose(output, expected.expand_as(output), atol=1e-6)
