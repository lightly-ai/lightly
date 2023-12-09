import pytest
import torch
from PIL import Image

from lightly.transforms.mmcr_transform import MMCRTransform


def test_raise_value_error() -> None:
    with pytest.raises(ValueError):
        MMCRTransform(k=0)


def test_num_views() -> None:
    multi_view_transform = MMCRTransform(k=3)
    assert len(multi_view_transform.transforms) == 3


def test_multi_view_on_tensor() -> None:
    multi_view_transform = MMCRTransform(k=3)
    sample = torch.rand(3, 100, 100)
    output = multi_view_transform(sample)
    assert len(output) == 3


def test_multi_view_on_pil_image() -> None:
    multi_view_transform = MMCRTransform(k=3)
    sample = Image.new("RGB", (100, 100))
    output = multi_view_transform(sample)
    assert len(output) == 3
