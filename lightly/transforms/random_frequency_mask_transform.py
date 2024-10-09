from typing import Tuple

import numpy as np
import torch
from torch import Tensor


class RandomFrequencyMaskTransform:
    """2D Random Frequency Mask Transformation.

    This transformation applies a binary mask on the fourier transform,
    across all channels. A proportion of k frequencies are set to 0 with this.

    Input
        - Tensor: RFFT of a 2D Image (C, H, W) C-> No. of Channels
    Output
        - Tensor: The masked RFFT of the image

    """

    def __init__(self, k: Tuple[float, float] = (0.01, 0.1)) -> None:
        self.k = k

    def __call__(self, fft_image: Tensor) -> Tensor:
        k = np.random.uniform(low=self.k[0], high=self.k[1])

        # Every mask for every channel will have same frequencies being turned off i.e. being set to zero
        mask = (
            torch.rand(fft_image.shape[1:], device=fft_image.device) > k
        )  # mask_type: (H, W)

        # Do not mask zero frequency mode to retain majority of the semantic information.
        # Please refer https://arxiv.org/abs/2312.02205
        mask[0, 0] = 1

        # Adding channel dimension
        mask = mask.unsqueeze(0)

        masked_frequency_spectrum_image = fft_image * mask

        return masked_frequency_spectrum_image
