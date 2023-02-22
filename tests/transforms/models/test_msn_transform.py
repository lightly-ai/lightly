from PIL import Image

from lightly.transforms.models.msn_transform import MSNTransform, MSNViewTransform


def test_view_on_pil_image():
    single_view_transform = MSNViewTransform(crop_size=32)
    sample = Image.new("RGB", (100, 100))
    output = single_view_transform(sample)
    assert output.shape == (3, 32, 32)


def test_multi_view_on_pil_image():
    multi_view_transform = MSNTransform(random_size=32, focal_size=8)
    sample = Image.new("RGB", (100, 100))
    output = multi_view_transform(sample)
    assert len(output) == 12
    for output_img in output:
        assert output_img.shape == (3, 32, 32) or output_img.shape == (3, 8, 8)
