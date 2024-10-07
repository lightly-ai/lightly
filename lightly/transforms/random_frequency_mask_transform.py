from typing import Tuple

import numpy as np
import torch
from torch import Tensor


class RFMTransform:
    """2D Random Frequency Mask Transformation.

    This transformation applies a binary mask on the fourier transform,
    across all channels. k% of frequencies are set to 0 with this.
    k ranges [0.01, 0.1)

    Input
        - Tensor: RFFT of a 2D Image (C, H, W) C-> No. of Channels
    Output
        - Tensor: The masked RFFT of the image

    """

    def __call__(self, fft_image: Tensor) -> Tensor:
        k = np.random.uniform(0.01, 0.1)
        # Mask: (C, H, W)
        mask = torch.ones_like(fft_image)

        total_frequencies = torch.numel(fft_image[0])
        num_frequencies_zeroed = int(total_frequencies * k)
        zero_frequency_idxs = torch.randperm(total_frequencies)[:num_frequencies_zeroed]

        # Every mask for every channel will have same frequencies being turned off i.e. being set to zero
        for c in range(mask.size(dim=0)):
            mask[c].view(-1)[zero_frequency_idxs] = 0
            # To retain majority of the semantic information. Please refer https://arxiv.org/abs/2312.02205
            mask[c][0][0] = 1

        masked_frequency_spectrum_image = fft_image * mask

        return masked_frequency_spectrum_image
