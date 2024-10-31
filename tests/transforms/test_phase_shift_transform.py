import torch

from lightly.transforms import IRFFT2DTransform, PhaseShiftTransform, RFFT2DTransform


# Testing function image -> RFFT -> PhaseShift.
# Compare shapes of source and result.
def test() -> None:
    image = torch.randn(3, 64, 64)

    rfftTransform = RFFT2DTransform()
    rfft = rfftTransform(image)

    phaseShiftTf_1 = PhaseShiftTransform()
    rescaled_rfft_1 = phaseShiftTf_1(rfft)

    phaseShiftTf_2 = PhaseShiftTransform(range=(1.0, 2.0))
    rescaled_rfft_2 = phaseShiftTf_2(rfft)

    assert rescaled_rfft_1.shape == rfft.shape
    assert rescaled_rfft_2.shape == rfft.shape
