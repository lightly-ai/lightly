""" Random Rotation """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import numpy as np
from torchvision.transforms import functional as TF
import torchvision.transforms as T
from typing import Tuple, Union


class RandomRotate(object):
    """Implementation of random rotation.

    Randomly rotates an input image by a fixed angle. By default, we rotate
    the image by 90 degrees with a probability of 50%.

    This augmentation can be very useful for rotation invariant images such as
    in medical imaging or satellite imaginary.

    Attributes:
        prob:
            Probability with which image is rotated.
        angle:
            Angle by which the image is rotated. We recommend multiples of 90
            to prevent rasterization artifacts. If you pick numbers like
            90, 180, 270 the tensor will be rotated without introducing 
            any artifacts.
    
    """

    def __init__(self, prob: float = 0.5, angle: int = 90):
        self.prob = prob
        self.angle = angle

    def __call__(self, sample):
        """Rotates the images with a given probability.

        Args:
            sample:
                PIL image which will be rotated.
        
        Returns:
            Rotated image or original image.

        """
        prob = np.random.random_sample()
        if prob < self.prob:
            sample = TF.rotate(sample, self.angle)
        return sample

class RandomRotationTransform(object):
    def __init__(self,
        rr_prob: float,
        rr_degrees: Union[None, float, Tuple[float, float]],
    ) -> Union[RandomRotate, T.RandomApply]:
        self.rr_prob = rr_prob,
        self.rr_degrees = rr_degrees

    def __call__(self, sample):
        if self.rr_degrees is None:
            # Random rotation by 90 degrees.
            rr = RandomRotate(prob=self.rr_prob[0], angle=90)
            return rr(sample)
        else:
            # Random rotation with random angle defined by rr_degrees.
            rr = T.RandomApply([T.RandomRotation(degrees=self.rr_degrees)], p=self.rr_prob)
            return rr(sample)