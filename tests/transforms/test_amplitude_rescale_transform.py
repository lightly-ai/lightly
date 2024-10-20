import numpy as np
import torch

from lightly.transforms import (
    AmplitudeRescaleTranform,
    IRFFT2DTransform,
    RFFT2DTransform,
)


# Testing function image -> FFT -> AmplitudeRescale.
# Compare shapes of source and result.
def test() -> None:
    image = torch.randn(3, 64, 64)

    rfftTransform = RFFT2DTransform()
    rfft = rfftTransform(image)

    ampRescaleTf_1 = AmplitudeRescaleTranform()
    rescaled_rfft_1 = ampRescaleTf_1(rfft)

    ampRescaleTf_2 = AmplitudeRescaleTranform(range=(1.0, 2.0))
    rescaled_rfft_2 = ampRescaleTf_2(rfft)

    assert rescaled_rfft_1.shape == rfft.shape
    assert rescaled_rfft_2.shape == rfft.shape
