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

    def __init__(self, range: Tuple[float, float] = (0.4, 0.7), include_negatives: bool = False, sign_probability: float = 0.5) -> None:
        self.dist = Uniform(range[0], range[1])
        self.include_negatives = include_negatives
        if include_negatives:
            self.sign_dist = Bernoulli(sign_probability)

    def __call__(self, freq_image: Tensor) -> Tensor:
        # Calculate amplitude and phase
        amplitude = torch.sqrt(freq_image.real**2 + freq_image.imag**2)
        phase = torch.atan2(freq_image.imag, freq_image.real)

        # Sample a random phase shift θ
        theta = self.dist.sample().to(freq_image.device)

        if self.include_negatives:
            # Determine sign for shift: +θ or -θ
            sign = self.sign_dist.sample().to(freq_image.device)
            # Apply random sign directly to theta
            theta = torch.where(sign == 1, theta, -theta)
        
        # Adjust the phase
        phase_shifted = torch.add(phase, theta)

        # Recreate the complex spectrum with adjusted phase
        real = amplitude * torch.cos(phase_shifted)
        imag = amplitude * torch.sin(phase_shifted)
        output = torch.complex(real, imag)

        return output
