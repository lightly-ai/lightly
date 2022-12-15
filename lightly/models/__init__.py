"""The lightly.models package provides model implementations.

Note that the high-level building blocks will be deprecated with 
lightly version 1.3.0. Instead, use low-level building blocks to build the
models yourself.

Example implementations for all models can be found here:
`Model Examples <https://docs.lightly.ai/examples/models.html>`_

The package contains an implementation of the commonly used ResNet and
adaptations of the architecture which make self-supervised learning simpler.

The package also hosts the Lightly model zoo - a list of downloadable ResNet
checkpoints.

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.models.resnet import ResNetGenerator
from lightly.models.barlowtwins import BarlowTwins
from lightly.models.simclr import SimCLR
from lightly.models.simsiam import SimSiam
from lightly.models.swav import SwaV
from lightly.models.byol import BYOL
from lightly.models.moco import MoCo
from lightly.models.nnclr import NNCLR
from lightly.models.zoo import ZOO
from lightly.models.zoo import checkpoints

from lightly.models import utils