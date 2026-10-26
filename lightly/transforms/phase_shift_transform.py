from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Uniform
from torch.distributions.bernoulli import Bernoulli


class PhaseShiftTransform:
    """Implementation of phase shifting transformation.


    Applies a random phase shift `theta` (positive or negative) to the Fourier spectrum (`freq_image`) of the image and returns the transformed spectrum.

    Attributes:
        dist:
            A uniform distribution in the range `[p, q)` from which the magnitude of the
            phase shift `theta` is selected.
        include_negatives:
            A flag indicating whether negative values of `theta` should be included.
            If `True`, both positive and negative shifts are applied.
        sign_dist:
            A Bernoulli distribution used to decide the sign of `theta`, based on a
            given probability `sign_probability`, if negative values are included.
    """

    def __init__(
        self,
        range: Tuple[float, float] = (0.4, 0.7),
        include_negatives: bool = False,
        sign_probability: float = 0.5,
    ) -> None:
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
        phase_shifted = phase + theta

        # Recreate the complex spectrum with adjusted phase
        real = amplitude * torch.cos(phase_shifted)
        imag = amplitude * torch.sin(phase_shifted)
        output = torch.complex(real, imag)

        return output
