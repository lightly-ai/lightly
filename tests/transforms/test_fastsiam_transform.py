from PIL import Image

from lightly.transforms.fast_siam_transform import FastSiamTransform


def test_multi_view_on_pil_image() -> None:
    multi_view_transform = FastSiamTransform(num_views=3, input_size=32)
    sample = Image.new("RGB", (100, 100))
    output = multi_view_transform(sample)
    assert len(output) == 3
    assert output[0].shape == (3, 32, 32)
    assert output[1].shape == (3, 32, 32)
    assert output[2].shape == (3, 32, 32)
