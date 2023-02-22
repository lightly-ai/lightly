from PIL import Image

from lightly.transforms.models.swav_transform import SwaVTransform, SwaVViewTransform


def test_view_on_pil_image():
    single_view_transform = SwaVViewTransform()
    sample = Image.new("RGB", (100, 100))
    output = single_view_transform(sample)
    assert output.shape == (3, 100, 100)


def test_multi_view_on_pil_image():
    multi_view_transform = SwaVTransform(crop_sizes=(32, 8))
    sample = Image.new("RGB", (100, 100))
    output = multi_view_transform(sample)
    assert len(output) == 8
    for output_img in output:
        assert output_img.shape == (3, 32, 32) or output_img.shape == (3, 8, 8)
