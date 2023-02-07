""" Gaussian Blur """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import numpy as np
from PIL import ImageFilter
from typing import Tuple


class GaussianBlur(object):
    """Implementation of random Gaussian blur.

    Utilizes the built-in ImageFilter method from PIL to apply a Gaussian 
    blur to the input image with a certain probability. The blur is further
    randomized by sampling uniformly the values of the standard deviation of
    the Gaussian kernel.

    Attributes:
        kernel_size:
            Mean kernel size for the Gaussian blur.
        prob:
            Probability with which the blur is applied.
        scale:
            Old unused parameters kept for compatibility
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled

    """

    def __init__(self, kernel_size: float, prob: float = 0.5,
                 scale: float = 0.2, sigmas: Tuple[float, float] = (.2, 2)):
        self.prob = prob
        self.sigmas = sigmas

    def __call__(self, sample):
        """Blurs the image with a given probability.

        Args:
            sample:
                PIL image to which blur will be applied.
        
        Returns:
            Blurred image or original image.
        """
        prob = np.random.random_sample()
        if prob < self.prob:
            # choose randomized std for Gaussian filtering
            sigma = np.random.uniform(self.sigmas[0], self.sigmas[1])
            return sample.filter(ImageFilter.GaussianBlur(radius=sigma))
        # return original image
        return sample
