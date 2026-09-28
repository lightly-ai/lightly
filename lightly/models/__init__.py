"""The lightly.models package provides model implementations.

Example implementations for all models can be found here:
`Model Examples <https://docs.lightly.ai/self-supervised-learning/examples/models.html>`_

The package contains an implementation of the commonly used ResNet and
adaptations of the architecture which make self-supervised learning simpler.

The package also hosts the Lightly model zoo - a list of downloadable ResNet
checkpoints.

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.models import utils
from lightly.models.resnet import ResNetGenerator
from lightly.models.zoo import ZOO, checkpoints
