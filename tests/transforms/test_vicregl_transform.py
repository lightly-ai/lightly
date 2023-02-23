from PIL import Image
from lightly.transforms.vicregl_transform import (
    VICRegLTransform,
    VICRegLViewTransform,
)


def test_view_on_pil_image():
    single_view_transform = VICRegLViewTransform()
    sample = Image.new("RGB", (100, 100))
    output = single_view_transform(sample)
    assert output.shape == (3, 100, 100)


def test_multi_view_on_pil_image():
    multi_view_transform = VICRegLTransform(
        global_crop_size=32, local_crop_size=8, global_grid_size=4, local_grid_size=2
    )
    sample = Image.new("RGB", (100, 100))
    output = multi_view_transform(sample)
    assert len(output) == 4
    assert output[0].shape == (3, 32, 32)
    assert output[1].shape == (3, 8, 8)
    assert output[2].shape == (4, 4, 2)
    assert output[3].shape == (2, 2, 2)
