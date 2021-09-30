"""The lightly.models package provides model implementations.

*Note that the high-level building blocks will be deprecated with 
lightly version 1.2.0. Instead, use low-level building blocks to build the
models yourself. 
Have a look at the benchmark code to see a reference implementation:*
`lightly benchmarks <https://github.com/lightly-ai/lightly/tree/master/docs/source/getting_started/benchmarks>`_

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
from lightly.models.byol import BYOL
from lightly.models.moco import MoCo
from lightly.models.nnclr import NNCLR
from lightly.models.zoo import ZOO
from lightly.models.zoo import checkpoints

from lightly.models import utils