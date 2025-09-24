from typing import Union

import torch
from torch import Tensor


class RFFT2DTransform:
    """2D Fast Fourier Transform (RFFT2D) Transformation.

    This transformation applies the 2D Fast Fourier Transform (RFFT2D)
    to an image, converting it from the spatial domain to the frequency domain.

    Input:
        - Tensor of shape (C, H, W), where C is the number of channels.

    Output:
        - Tensor of shape (C, H, W) in the frequency domain, where C is the number of channels.
    """

    def __call__(self, image: Tensor) -> Tensor:
        """Applies the 2D Fast Fourier Transform (RFFT2D) to the input image.

        Args:
            image: Input image as a Tensor of shape (C, H, W).

        Returns:
            Tensor: The image in the frequency domain after applying RFFT2D, of shape (C, H, W).
        """

        rfft_image: Tensor = torch.fft.rfft2(image)
        return rfft_image
