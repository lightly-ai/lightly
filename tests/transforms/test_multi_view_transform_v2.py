import unittest

import pytest
from PIL import Image

from lightly.utils.dependency import torchvision_transforms_v2_available

if not torchvision_transforms_v2_available():
    pytest.skip("torchvision.transforms.v2 not available", allow_module_level=True)
import torch
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import Mask

from lightly.transforms.multi_view_transform import MultiViewTransform


def test_multi_view_on_pil_image() -> None:
    multi_view_transform = MultiViewTransform(
        [
            T.RandomHorizontalFlip(p=0.1),
            T.RandomVerticalFlip(p=0.5),
            T.RandomGrayscale(p=0.3),
        ]
    )
    sample = {"img": Image.new("RGB", (10, 10)), "mask": Mask(torch.zeros(10, 10))}
    output = multi_view_transform(sample)
    assert len(output) == 3
