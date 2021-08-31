""" SimCLR Model """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn

from lightly.models.base_embedding_model import BaseEmbeddingModel
from lightly.models.modules import SimCLRProjectionHead


class SimCLR(BaseEmbeddingModel):
    """Implementation of the SimCLR[0] architecture

    Recommended loss: :py:class:`lightly.loss.ntx_ent_loss.NTXentLoss`

    [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709

    Attributes:
        backbone:
            Backbone model to extract features from images.
        num_ftrs:
            Dimension of the embedding (before the projection head).
        out_dim:
            Dimension of the output (after the projection head).

    """

    def __init__(self,
                 backbone: nn.Module,
                 num_ftrs: int = 32,
                 out_dim: int = 128):

        projection_head = SimCLRProjectionHead(num_ftrs, num_ftrs, out_dim)
        super(SimCLR, self).__init__(backbone=backbone, head=projection_head)

        # Allow pretrained models to keep working
        # Otherwise the test of the CLI on a real dataset (on the setup.py) returns
        """
        RuntimeError: Error(s) in loading state_dict for SimCLR: 32
	    Missing key(s) in state_dict: "head.layers.0.weight", "head.layers.0.bias", 
	    "head.layers.2.weight", "head.layers.2.bias". 33
	    Unexpected key(s) in state_dict: "projection_head.layers.0.weight", "projection_head.layers.0.bias", 
	    "projection_head.layers.2.weight", "projection_head.layers.2.bias".
        """
        self.projection_head = self.head
