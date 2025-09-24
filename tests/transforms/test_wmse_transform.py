import pytest
from PIL import Image

from lightly.transforms.wmse_transform import WMSETransform


def test_raise_value_error() -> None:
    with pytest.raises(ValueError):
        WMSETransform(num_samples=0)


def test_num_views() -> None:
    multi_view_transform = WMSETransform(num_samples=3)
    assert len(multi_view_transform.transforms) == 3


def test_multi_view_on_pil_image() -> None:
    multi_view_transform = WMSETransform(num_samples=3)
    sample = Image.new("RGB", (100, 100))
    output = multi_view_transform(sample)
    assert len(output) == 3
