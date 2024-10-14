from typing import Tuple

import torch
import torch.fft
from torch import Tensor

from lightly.transforms.irfft2d_transform import IRFFT2DTransform
from lightly.transforms.rfft2d_transform import RFFT2DTransform


class GaussianMixtureMask:
    """Applies a Gaussian Mixture Mask in the Fourier domain to a single-channel image.

    The mask is created using random Gaussian kernels, which are applied in
    the frequency domain via RFFT2D, and then the IRFFT2D is used to return
    to the spatial domain. The transformation is applied to each image channel separately.

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

        self.rfft2d_transform = RFFT2DTransform()

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
            Tensor: A 2D Gaussian kernel.
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
        self, freq_image: Tensor, num_gaussians: int, std: Tuple[int, int]
    ) -> Tensor:
        """Applies the Gaussian mixture mask to a frequency-domain image.

        Args:
            freq_image: Tensor representing the frequency-domain image of shape (C, H, W//2+1).
            num_gaussians: Number of Gaussian kernels to generate in the mask.
            std: Tuple specifying the standard deviation range for the Gaussians.

        Returns:
            Tensor: Image after applying the Gaussian mixture mask.
        """
        image_size = freq_image.shape[1:]
        original_height = image_size[0]
        original_width = 2 * (image_size[1] - 1)

        original_shape = (original_height, original_width)

        self.irfft2d_transform = IRFFT2DTransform(original_shape)

        size = freq_image[0].shape

        mask = freq_image.new_ones(freq_image.shape)

        for _ in range(num_gaussians):
            u0 = torch.randint(0, size[0], (1,), device=freq_image.device)
            v0 = torch.randint(0, size[1], (1,), device=freq_image.device)
            center = torch.tensor((u0, v0), device=freq_image.device)
            sigma = torch.rand(2, device=freq_image.device) * (std[1] - std[0]) + std[0]

            g_kernel = self.gaussian_kernel((size[0], size[1]), sigma, center)
            mask -= g_kernel

        filtered_freq_image = freq_image * mask
        filtered_image = self.irfft2d_transform(filtered_freq_image).abs()
        return filtered_image

    def __call__(self, freq_image: Tensor) -> Tensor:
        """Applies the Gaussian mixture mask transformation to the input frequency-domain image.

        Args:
            freq_image: Tensor representing a frequency-domain image of shape (C, H, W//2+1).

        Returns:
            Tensor: The transformed image after applying the Gaussian mixture mask.
        """
        transformed_channel: Tensor = self.apply_gaussian_mixture_mask(
            freq_image, self.num_gaussians, self.std_range
        )
        return transformed_channel
