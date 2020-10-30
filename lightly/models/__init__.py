"""The lightly.models package provides model implementations.

The package contains an implementation of the commonly used ResNet and 
adaptations of the architecture which make self-supervised learning simpler.

The package also hosts the Lightly model zoo - a list of downloadable ResNet 
checkpoints.

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.models.resnet import ResNetGenerator
from lightly.models.simclr import ResNetSimCLR
from lightly.models.moco import ResNetMoCo
from lightly.models.zoo import ZOO
from lightly.models.zoo import checkpoints
