import unittest

import torchvision.transforms as T
from PIL import Image

from lightly.transforms.multi_view_transform import MultiViewTransform


def test_multi_view_on_pil_image():
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
