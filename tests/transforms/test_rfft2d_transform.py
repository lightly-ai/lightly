import torch

from lightly.transforms import RFFT2DTransform


def test() -> None:
    transform = RFFT2DTransform()
    image = torch.rand(3, 32, 32)
    output = transform(image)
    assert output.shape == (3, 32, 17)
