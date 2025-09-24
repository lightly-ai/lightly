from PIL import Image

from lightly.transforms.mae_transform import MAETransform


def test_multi_view_on_pil_image() -> None:
    multi_view_transform = MAETransform(input_size=32)
    sample = Image.new("RGB", (100, 100))
    output = multi_view_transform(sample)
    assert len(output) == 1
    assert output[0].shape == (3, 32, 32)
