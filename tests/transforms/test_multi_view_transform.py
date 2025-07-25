import unittest

from PIL import Image

from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T


def test_multi_view_on_pil_image() -> None:
    multi_view_transform = MultiViewTransform(
        [
            T.RandomHorizontalFlip(p=0.1),
            T.RandomVerticalFlip(p=0.5),
            T.RandomGrayscale(p=0.3),
        ]
    )
    sample = Image.new("RGB", (10, 10))
    output = multi_view_transform(sample)
    assert len(output) == 3
