""" Random Rotation """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import numpy as np
from torchvision.transforms import functional as TF


class RandomRotate(object):
    """Implementation of random rotation.

    Randomly rotates an input image by an angle.

    Attributes:
        prob:
            Probability with which image is rotated.
        angle:
            Angle by which the image is rotated.
    
    """

    def __init__(self, prob: float = 0.5, angle: int = 90):
        self.prob = prob
        self.angle = 90

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
