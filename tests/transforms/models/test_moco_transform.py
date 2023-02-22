from PIL import Image
import torch

from lightly.transforms.models.moco_transform import MoCoTransform


def test_multi_view_on_pil_image():
    multi_view_transform = MoCoTransform(input_size=32)
    sample = Image.new("RGB", (100, 100))
    output = multi_view_transform(sample)
    assert len(output) == 2
    assert output[0].shape == (3, 32, 32)
    assert output[1].shape == (3, 32, 32)
