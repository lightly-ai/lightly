"""The lightly.transforms package provides additional augmentations.

    Contains implementations of Gaussian blur and random rotations which are
    not part of torchvisions transforms.

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.transforms.dino_transform import DINOTransform, DINOViewTransform
from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.jigsaw import Jigsaw
from lightly.transforms.mae_transform import MAETransform
from lightly.transforms.moco_transform import MoCoV1Transform, MoCoV2Transform
from lightly.transforms.msn_transform import MSNTransform, MSNViewTransform
from lightly.transforms.pirl_transform import PIRLTransform
from lightly.transforms.rotation import (
    RandomRotate,
    RandomRotateDegrees,
    random_rotation_transform,
)
from lightly.transforms.simclr_transform import SimCLRTransform, SimCLRViewTransform
from lightly.transforms.smog_transform import SMoGTransform, SmoGViewTransform
from lightly.transforms.solarize import RandomSolarization
from lightly.transforms.swav_transform import SwaVTransform, SwaVViewTransform
from lightly.transforms.vicreg_transform import VICRegTransform, VICRegViewTransform
from lightly.transforms.vicregl_transform import VICRegLTransform, VICRegLViewTransform
