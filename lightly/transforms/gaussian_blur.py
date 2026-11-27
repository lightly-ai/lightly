# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import math
from typing import Tuple, Union

import numpy as np
from PIL.Image import Image
from torch import Tensor

from lightly.transforms.torchvision_v2_compatibility import functional as F


class GaussianBlur:
    """Implementation of random Gaussian blur.

    Utilizes torchvision's functional `gaussian_blur` to apply a Gaussian blur
    to the input image or tensor with a certain probability. The blur is further
    randomized by sampling uniformly the standard deviation (sigma) of the
    Gaussian kernel.

    Attributes:
        prob:
            Probability with which the blur is applied.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.

    """

    def __init__(
        self,
        *,
        prob: float = 0.5,
        sigmas: Tuple[float, float] = (0.2, 2),
    ):
        self.prob = prob
        self.sigmas = sigmas

    def __call__(self, sample: Union[Tensor, Image]) -> Union[Tensor, Image]:
        """Blurs the image with a given probability.

        Args:
            sample:
                PIL image or tensor to which blur will be applied.

        Returns:
            Blurred image or original image.
        """
        prob = np.random.random_sample()
        if prob >= self.prob:
            return sample

        # Choose randomized std for Gaussian filtering (preserving #1052 sigma semantics).
        sigma = float(np.random.uniform(self.sigmas[0], self.sigmas[1]))
        if sigma <= 0.0:
            return sample

        # Use an odd kernel covering +/- 3*sigma, clamped per axis so torchvision's
        # reflect padding (kernel_size // 2) remains strictly smaller than the image
        # dimension even for tiny inputs (e.g. 1x1).
        if isinstance(sample, Tensor):
            height, width = sample.shape[-2], sample.shape[-1]
        else:
            width, height = sample.size

        base_kernel = max(1, 2 * math.ceil(3.0 * sigma) + 1)
        kernel_w = min(base_kernel, 2 * width - 1)
        kernel_h = min(base_kernel, 2 * height - 1)

        # torchvision functional.gaussian_blur expects kernel_size as [kernel_x, kernel_y].
        blurred: Union[Tensor, Image] = F.gaussian_blur(
            sample,
            kernel_size=[kernel_w, kernel_h],
            sigma=[sigma, sigma],
        )
        return blurred
