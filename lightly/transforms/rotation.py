# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import Tuple, Union

import numpy as np
import torchvision.transforms as T
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import functional as TF


class RandomRotate:
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

    def __call__(self, image: Union[Image, Tensor]) -> Union[Image, Tensor]:
        """Rotates the image with a given probability.

        Args:
            image:
                PIL image or tensor which will be rotated.

        Returns:
            Rotated image or original image.

        """
        prob = np.random.random_sample()
        if prob < self.prob:
            image = TF.rotate(image, self.angle)
        return image


class RandomRotateDegrees:
    """Random rotate image between two rotation angles with a random probability.

    Attributes:
        prob:
            Probability with which image is rotated.
        degrees:
            Range of degrees to select from. If degrees is a number instead of a sequence like (min, max),
            the range of degrees will be (-degrees, +degrees). The image is rotated counter-clockwise with
            a random angle in the (min, max) range or in the (-degrees, +degrees) range.

    """

    def __init__(self, prob: float, degrees: Union[float, Tuple[float, float]]):
        self.transform = T.RandomApply([T.RandomRotation(degrees=degrees)], p=prob)

    def __call__(self, image: Union[Image, Tensor]) -> Union[Image, Tensor]:
        """Rotates the images with a given probability.

        Args:
            image:
                PIL image or tensor which will be rotated.

        Returns:
            Rotated image or original image.

        """
        return self.transform(image)


def random_rotation_transform(
    rr_prob: float,
    rr_degrees: Union[None, float, Tuple[float, float]],
) -> Union[RandomRotate, T.RandomApply]:
    if rr_degrees is None:
        # Random rotation by 90 degrees.
        return RandomRotate(prob=rr_prob, angle=90)
    else:
        # Random rotation with random angle defined by rr_degrees.
        return RandomRotateDegrees(prob=rr_prob, degrees=rr_degrees)
