import torch

from lightly.transforms import GaussianMixtureMask


def test() -> None:
    transform = GaussianMixtureMask(20, (10, 15))
    image = torch.rand(3, 32, 17)
    output = transform(image)
    assert output.shape == image.shape
