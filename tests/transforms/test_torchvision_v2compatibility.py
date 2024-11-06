import numpy as np
import torch
from PIL import Image
from torch import Tensor

from lightly.transforms import ToTensor


def test_ToTensor() -> None:
    img_np = np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)
    img_pil = Image.fromarray(img_np)
    img_tens = ToTensor()(img_pil)
    assert isinstance(img_tens, Tensor)
    assert img_tens.shape == (3, 20, 30)
    assert img_tens.dtype == torch.float32
    assert img_tens.max() <= 1.0 and img_tens.min() >= 0.0
