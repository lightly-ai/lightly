from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Uniform
from torch.distributions.bernoulli import Bernoulli


class PhaseShiftTransform:
    """Implementation of phase shifting transformation

    Adds a random phase shift `theta` (positive or negative),
    to all components of the Fourier spectrum (`freq_image`) of the image and return it.

    Attributes:
        dist:
            Uniform distribution in `[p, q)` from which the magnitude of phase shift will be selected.
    """

    def __init__(self, range: Tuple[float, float] = (0.4, 0.7)) -> None:
        self.dist = Uniform(range[0], range[1])
        self.sign_dist = Bernoulli(0.5)

    def __call__(self, freq_image: Tensor) -> Tensor:
        # Calculate amplitude and phase
        amplitude = torch.sqrt(freq_image.real**2 + freq_image.imag**2)
        phase = torch.atan2(freq_image.imag, freq_image.real)

        # Sample a random phase shift θ for each channel and set shape
        theta = self.dist.sample(freq_image.shape[1:]).to(freq_image.device)

        # Determine sign for each shift: +θ or -θ
        signs = self.sign_dist.sample(freq_image.shape[1:]).to(freq_image.device)
        theta = torch.where(
            signs == 1, theta, -theta
        )  # Apply random sign directly to theta

        # Adjust the phase
        phase_shifted = phase + theta

        # Recreate the complex spectrum with adjusted phase
        real = amplitude * torch.cos(phase_shifted)
        imag = amplitude * torch.sin(phase_shifted)
        output = torch.complex(real, imag)

        return output
