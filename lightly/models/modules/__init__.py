"""The lightly.models.modules package provides reusable modules.

This package contains reusable modules such as the NNmemoryBankModule which
can be combined with any lightly model.

"""

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.models.modules.heads import BarlowTwinsProjectionHead
from lightly.models.modules.heads import BYOLProjectionHead
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.modules.heads import NNCLRProjectionHead
from lightly.models.modules.heads import NNCLRPredictionHead
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.models.modules.heads import SimSiamProjectionHead
from lightly.models.modules.heads import SimSiamPredictionHead
from lightly.models.modules.nn_memory_bank import NNMemoryBankModule
