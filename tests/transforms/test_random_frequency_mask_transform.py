import torch

from lightly.transforms import RandomFrequencyMaskTransform, RFFT2DTransform


def test() -> None:
    rfm_transform = RandomFrequencyMaskTransform()
    rfft2d_transform = RFFT2DTransform()
    image = torch.randn(3, 64, 64)
    fft_image = rfft2d_transform(image)
    transformed_image = rfm_transform(fft_image)

    assert transformed_image.shape == fft_image.shape
