"""The lightly.transforms package transforms for various self-supervised learning
methods.

It also contains some additional transforms that are not part of torchvisions
transforms.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.transforms.aim_transform import AIMTransform
from lightly.transforms.amplitude_rescale_transform import AmplitudeRescaleTransform
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
from lightly.transforms.densecl_transform import DenseCLTransform
from lightly.transforms.dino_transform import DINOTransform, DINOViewTransform
from lightly.transforms.fast_siam_transform import FastSiamTransform
from lightly.transforms.fda_transform import (
    FDATransform,
    FDAView1Transform,
    FDAView2Transform,
)
from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.gaussian_mixture_masks_transform import GaussianMixtureMask
from lightly.transforms.irfft2d_transform import IRFFT2DTransform
from lightly.transforms.jigsaw import Jigsaw
from lightly.transforms.mae_transform import MAETransform
from lightly.transforms.mmcr_transform import MMCRTransform
from lightly.transforms.moco_transform import MoCoV1Transform, MoCoV2Transform
from lightly.transforms.msn_transform import MSNTransform, MSNViewTransform
from lightly.transforms.phase_shift_transform import PhaseShiftTransform
from lightly.transforms.pirl_transform import PIRLTransform
from lightly.transforms.random_frequency_mask_transform import (
    RandomFrequencyMaskTransform,
)
from lightly.transforms.rfft2d_transform import RFFT2DTransform
from lightly.transforms.rotation import (
    RandomRotate,
    RandomRotateDegrees,
    random_rotation_transform,
)
from lightly.transforms.simclr_transform import SimCLRTransform, SimCLRViewTransform
from lightly.transforms.simsiam_transform import SimSiamTransform, SimSiamViewTransform
from lightly.transforms.smog_transform import SMoGTransform, SmoGViewTransform
from lightly.transforms.solarize import RandomSolarization
from lightly.transforms.swav_transform import SwaVTransform, SwaVViewTransform
from lightly.transforms.tico_transform import (
    TiCoTransform,
    TiCoView1Transform,
    TiCoView2Transform,
)
from lightly.transforms.torchvision_v2_compatibility import (
    ToTensor,
    torchvision_transforms,
)
from lightly.transforms.vicreg_transform import VICRegTransform, VICRegViewTransform
from lightly.transforms.vicregl_transform import VICRegLTransform, VICRegLViewTransform
from lightly.transforms.wmse_transform import WMSETransform
from lightly.utils.dependency import torchvision_transforms_v2_available

if torchvision_transforms_v2_available():
    from lightly.transforms.add_grid_transform import AddGridTransform
    from lightly.transforms.detcon_transform import (
        DetConSTransform,
        DetConSViewTransform,
    )
    from lightly.transforms.multi_view_transform_v2 import MultiViewTransformV2
