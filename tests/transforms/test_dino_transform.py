from PIL import Image

from lightly.transforms.dino_transform import DINOTransform, DINOViewTransform


def test_view_on_pil_image() -> None:
    single_view_transform = DINOViewTransform(crop_size=32)
    sample = Image.new("RGB", (100, 100))
    output = single_view_transform(sample)
    assert output.shape == (3, 32, 32)


def test_multi_view_on_pil_image() -> None:
    multi_view_transform = DINOTransform(global_crop_size=32, local_crop_size=8)
    sample = Image.new("RGB", (100, 100))
    output = multi_view_transform(sample)
    assert len(output) == 8
    # global views
    assert all(out.shape == (3, 32, 32) for out in output[:2])
    # local views
    assert all(out.shape == (3, 8, 8) for out in output[2:])
