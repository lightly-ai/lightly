import torch

from lightly.transforms import RFFT2DTransform, RFMTransform


def test() -> None:
    rfm_transform = RFMTransform()
    rfft2d_transform = RFFT2DTransform()
    image = torch.randn(3, 64, 64)
    fft_image = rfft2d_transform(image)
    transformed_image = rfm_transform(fft_image)

    assert transformed_image.shape == fft_image.shape
