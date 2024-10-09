from PIL import Image

from lightly.transforms.swav_transform import SwaVTransform, SwaVViewTransform

from .. import helpers


def test_view_on_pil_image() -> None:
    single_view_transform = SwaVViewTransform()
    sample = Image.new("RGB", (100, 100))
    output = single_view_transform(sample)
    assert output.shape == (3, 100, 100)


def test_multi_view_on_pil_image() -> None:
    multi_view_transform = SwaVTransform(crop_sizes=(32, 8))
    sample = Image.new("RGB", (100, 100))
    output = helpers.assert_list_tensor(multi_view_transform(sample))
    assert len(output) == 8
    assert all(out.shape == (3, 32, 32) for out in output[:2])
    assert all(out.shape == (3, 8, 8) for out in output[2:])
