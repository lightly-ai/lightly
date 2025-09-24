from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Uniform


class AmplitudeRescaleTransform:
    """Implementation of amplitude rescaling transformation.

    This transform will rescale the amplitude of the Fourier Spectrum (`freq_image`) of the image and return it.

    Attributes:
        dist:
            Uniform distribution in `[m, n)` from which the scaling value will be selected.
    """

    def __init__(self, range: Tuple[float, float] = (0.8, 1.75)) -> None:
        self.dist = Uniform(range[0], range[1])

    def __call__(self, freq_image: Tensor) -> Tensor:
        amplitude = torch.sqrt(freq_image.real**2 + freq_image.imag**2)

        phase = torch.atan2(freq_image.imag, freq_image.real)
        # p with shape (H, W)
        p = self.dist.sample(freq_image.shape[1:]).to(freq_image.device)
        # Unsqueeze to add channel dimension.
        amplitude *= p.unsqueeze(0)
        real = amplitude * torch.cos(phase)
        imag = amplitude * torch.sin(phase)
        output = torch.complex(real, imag)

        return output
