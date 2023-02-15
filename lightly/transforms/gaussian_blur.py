""" Gaussian Blur """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import numpy as np
from PIL import ImageFilter
from typing import Tuple, Union
from warnings import warn


class GaussianBlur(object):
    """Implementation of random Gaussian blur.

    Utilizes the built-in ImageFilter method from PIL to apply a Gaussian
    blur to the input image with a certain probability. The blur is further
    randomized by sampling uniformly the values of the standard deviation of
    the Gaussian kernel.

    Attributes:
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        prob:
            Probability with which the blur is applied.
        scale:
            Old argument. Value is deprecated in favor of sigmas. If set, the old behavior applies and `sigmas` is ignored.
            Used to scale the `kernel_size` of a factor of `kernel_scale`
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.

    """

    def __init__(
        self,
        kernel_size: Union[float, None] = None,
        prob: float = 0.5,
        scale: Union[float, None] = None,
        sigmas: Tuple[float, float] = (0.2, 2),
    ):
        if scale != None or kernel_size != None:
            warn(
                "The 'kernel_size' and 'scale' arguments of the GaussianBlur augmentation will be deprecated.  "
                "Please use the 'sigmas' parameter instead.",
                PendingDeprecationWarning,
            )
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
            # PIL GaussianBlur https://github.com/python-pillow/Pillow/blob/76478c6865c78af10bf48868345db2af92f86166/src/PIL/ImageFilter.py#L154 label the
            # sigma parameter of the gaussian filter as radius. Before, the radius of the patch was passed as the argument.
            # The issue was addressed here https://github.com/lightly-ai/lightly/issues/1051 and solved by AurelienGauffre.
            return sample.filter(ImageFilter.GaussianBlur(radius=sigma))
        # return original image
        return sample
