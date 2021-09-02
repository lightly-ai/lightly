"""The lightly.transforms package provides additional augmentations.

    Contains implementations of Gaussian blur and random rotations which are
    not part of torchvisions transforms.

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.rotation import RandomRotate
from lightly.transforms.solarize import RandomSolarization
from lightly.transforms.jigsaw import Jigsaw
