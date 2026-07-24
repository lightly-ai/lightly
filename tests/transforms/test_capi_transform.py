from __future__ import annotations

from PIL import Image

from lightly.transforms import CAPITransform


def test_view_on_pil_image() -> None:
    transform = CAPITransform(input_size=32)
    output = transform(Image.new("RGB", (100, 100)))
    assert len(output) == 1
    assert output[0].shape == (3, 32, 32)


def test_default_input_size() -> None:
    transform = CAPITransform()
    output = transform(Image.new("RGB", (256, 256)))
    assert output[0].shape == (3, 224, 224)
