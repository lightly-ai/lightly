"""The lightly.models.modules package provides reusable modules.

This package contains reusable modules such as the NNmemoryBankModule which
can be combined with any lightly model.

"""

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

from lightly import _timm_available, _timm_version, _torchvision_vit_available
from lightly.models.modules.heads import (
    BarlowTwinsProjectionHead,
    BYOLPredictionHead,
    BYOLProjectionHead,
    DINOProjectionHead,
    MoCoProjectionHead,
    NNCLRPredictionHead,
    NNCLRProjectionHead,
    SimCLRProjectionHead,
    SimSiamPredictionHead,
    SimSiamProjectionHead,
    SMoGPredictionHead,
    SMoGProjectionHead,
    SMoGPrototypes,
    SwaVProjectionHead,
    SwaVPrototypes,
)
from lightly.models.modules.nn_memory_bank import NNMemoryBankModule
from packaging.version import Version as _Version

if _torchvision_vit_available:
    # Requires torchvision >=0.12
    from lightly.models.modules.masked_autoencoder import (
        MAEBackbone,
        MAEDecoder,
        MAEEncoder,
    )
if _timm_available and _timm_version is not None and _timm_version >= _Version("0.9.9"):
    # Requires timm >= 0.9.9
    from lightly.models.modules.heads_timm import AIMPredictionHead
    from lightly.models.modules.masked_causal_vision_transformer import (
        MaskedCausalVisionTransformer,
    )
