"""The lightly.models.modules package provides reusable modules.

This package contains reusable modules such as the NNmemoryBankModule which
can be combined with any lightly model.

"""

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved


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
from lightly.utils import dependency as _dependency

if _dependency.torchvision_vit_available():
    # Requires torchvision >=0.12
    from lightly.models.modules.masked_autoencoder import (
        MAEBackbone,
        MAEDecoder,
        MAEEncoder,
    )
if _dependency.timm_vit_available():
    # Requires timm >= 0.9.9
    from lightly.models.modules.heads_timm import AIMPredictionHead
    from lightly.models.modules.masked_causal_vision_transformer import (
        MaskedCausalVisionTransformer,
    )
