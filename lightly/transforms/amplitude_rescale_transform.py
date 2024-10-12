from typing import Tuple

import numpy as np
import torch
from torch import Tensor


class AmplitudeRescaleTranform:
    """
    This transform will rescale the amplitude of the Fourier Spectrum (`input`) of the image and return it.
    The scaling value *p* will range within `[m, n)`
    ```
    img = torch.randn(3, 64, 64)

    rfft = lightly.transforms.RFFT2DTransform()
    rfft_img = rfft(img)

    art = AmplitudeRescaleTransform()
    rescaled_img = art(rfft_img)
    ```

    # Intial Arguments
        **range**: *Tuple of float_like*
        The low `m` and high `n` values such that **p belongs to [m, n)**.
    # Parameters:
        **input**: _torch.Tensor_
        The 2D Discrete Fourier Tranform of an Image.
    # Returns:
        **output**:_torch.Tensor_
        The Fourier spectrum of the 2D Image with rescaled Amplitude.
    """

    def __init__(self, range: Tuple[float, float] = (0.8, 1.75)) -> None:
        self.m = range[0]
        self.n = range[1]

    def __call__(self, input: Tensor) -> Tensor:
        p = np.random.uniform(self.m, self.n)

        output = input * p

        return output
