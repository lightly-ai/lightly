from PIL import Image
from lightly.transforms.models.vicreg_transform import (
    VICRegTransform,
    VICRegViewTransform,
)


def test_view_on_pil_image():
    single_view_transform = VICRegViewTransform(input_size=32)
    sample = Image.new("RGB", (100, 100))
    output = single_view_transform(sample)
    assert output.shape == (3, 32, 32)


def test_multi_view_on_pil_image():
    multi_view_transform = VICRegTransform(input_size=32)
    sample = Image.new("RGB", (100, 100))
    output = multi_view_transform(sample)
    assert len(output) == 2
    assert output[0].shape == (3, 32, 32)
    assert output[1].shape == (3, 32, 32)
