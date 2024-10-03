from typing import Tuple

import torch
from torch import Tensor


class IRFFT2DTransform:
    """Inverse 2D Fast Fourier Transform (IRFFT2D) Transformation.

    This transformation applies the inverse 2D Fast Fourier Transform (IRFFT2D)
    to an image in the frequency domain.

    Input:
        - Tensor of shape (C, H, W), where C is the number of channels.

    Output:
        - Tensor of shape (C, H, W), where C is the number of channels.
    """

    def __init__(self, shape: Tuple[int, int]):
        """
        Args:
            shape: The desired output shape (H, W) after applying the inverse FFT
        """
        self.shape = shape

    def __call__(self, freq_image: Tensor) -> Tensor:
        """Applies the inverse 2D Fast Fourier Transform (IRFFT2D) to the input tensor.

        Args:
            freq_image: A tensor in the frequency domain of shape (C, H, W).

        Returns:
            Tensor: Reconstructed image after applying IRFFT2D, of shape (C, H, W).
        """
        reconstructed_image: Tensor = torch.fft.irfft2(freq_image, s=self.shape)
        return reconstructed_image
