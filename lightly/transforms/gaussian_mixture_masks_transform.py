from typing import Tuple

import torch
import torch.fft
from torch import Tensor


class GaussianMixtureMask:
    """Applies a Gaussian Mixture Mask in the Fourier domain to an image.

    The mask is created using random Gaussian kernels, which are applied in
    the frequency domain.

    Attributes:
        num_gaussians: Number of Gaussian kernels to generate in the mixture mask.
        std_range: Tuple containing the minimum and maximum standard deviation for the Gaussians.
    """

    def __init__(
        self, num_gaussians: int = 20, std_range: Tuple[float, float] = (10, 15)
    ):
        """Initializes GaussianMixtureMasks with the given parameters.

        Args:
            num_gaussians: Number of Gaussian kernels to generate in the mixture mask.
            std_range: Tuple containing the minimum and maximum standard deviation for the Gaussians.
        """
        self.num_gaussians = num_gaussians
        self.std_range = std_range

    def gaussian_kernel(
        self, size: Tuple[int, int], sigma: Tensor, center: Tensor
    ) -> Tensor:
        """Generates a 2D Gaussian kernel.

        Args:
            size: Tuple specifying the dimensions of the Gaussian kernel (H, W).
            sigma: Tensor specifying the standard deviation of the Gaussian.
            center: Tensor specifying the center of the Gaussian kernel.

        Returns:
            A 2D Gaussian kernel tensor.
        """
        u, v = torch.meshgrid(torch.arange(0, size[0]), torch.arange(0, size[1]))
        u = u.to(sigma.device)
        v = v.to(sigma.device)
        u0, v0 = center
        gaussian = torch.exp(
            -((u - u0) ** 2 / (2 * sigma[0] ** 2) + (v - v0) ** 2 / (2 * sigma[1] ** 2))
        )

        return gaussian

    def apply_gaussian_mixture_mask(
        self, freq_image: Tensor, num_gaussians: int, std: Tuple[float, float]
    ) -> Tensor:
        """Applies the Gaussian mixture mask to a frequency-domain image.

        Args:
            freq_image: Tensor representing the frequency-domain image of shape (C, H, W//2+1).
            num_gaussians: Number of Gaussian kernels to generate in the mask.
            std: Tuple specifying the standard deviation range for the Gaussians.

        Returns:
            Image tensor in frequency domain after applying the Gaussian mixture mask.
        """
        (C, U, V) = freq_image.shape
        mask = freq_image.new_ones(freq_image.shape)

        for _ in range(num_gaussians):
            u0 = torch.randint(0, U, (1,), device=freq_image.device)
            v0 = torch.randint(0, V, (1,), device=freq_image.device)
            center = torch.tensor((u0, v0), device=freq_image.device)
            sigma = torch.rand(2, device=freq_image.device) * (std[1] - std[0]) + std[0]

            g_kernel = self.gaussian_kernel((U, V), sigma, center)
            mask *= 1 - g_kernel.unsqueeze(0)

        filtered_freq_image = freq_image * mask
        return filtered_freq_image

    def __call__(self, freq_image: Tensor) -> Tensor:
        """Applies the Gaussian mixture mask transformation to the input frequency-domain image.

        Args:
            freq_image: Tensor representing a frequency-domain image of shape (C, H, W//2+1).

        Returns:
            Image tensor in frequency domain after applying the Gaussian mixture mask.
        """
        return self.apply_gaussian_mixture_mask(
            freq_image, self.num_gaussians, self.std_range
        )
