from PIL import Image

from lightly.transforms.smog_transform import SMoGTransform, SmoGViewTransform


def test_view_on_pil_image():
    single_view_transform = SmoGViewTransform(crop_size=32)
    sample = Image.new("RGB", (100, 100))
    output = single_view_transform(sample)
    assert output.shape == (3, 32, 32)


def test_multi_view_on_pil_image():
    multi_view_transform = SMoGTransform(crop_sizes=(32, 8))
    sample = Image.new("RGB", (100, 100))
    output = multi_view_transform(sample)
    assert len(output) == 8
    assert all(out.shape == (3, 32, 32) for out in output[:4])
    assert all(out.shape == (3, 8, 8) for out in output[4:])
