import torch

from lightly.transforms import GaussianMixtureMasks


def test() -> None:
    transform = GaussianMixtureMasks(20, (10, 15))
    image = torch.rand(3, 32, 32)
    output = transform(image)
    assert output.shape == (3, 32, 32)
