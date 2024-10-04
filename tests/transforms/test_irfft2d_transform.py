import torch

from lightly.transforms import IRFFT2DTransform


def test() -> None:
    transform = IRFFT2DTransform((32, 32))
    image = torch.rand(3, 32, 17)
    output = transform(image)
    assert output.shape == (3, 32, 32)
