""" SimCLR Model """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import warnings

import torch
import torch.nn as nn

from lightly.models.modules import SimCLRProjectionHead


class _SimCLR(nn.Module):
    """Implementation of the SimCLR architecture for using the checkpoint

        Only used by the CLI for creating and loading checkpoints
    """

    def __init__(self, backbone: nn.Module, num_ftrs: int = 32,
                 out_dim: int = 128):

        super(SimCLR, self).__init__()

        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(num_ftrs, num_ftrs,
                                                    out_dim)


    def forward(self, x0: torch.Tensor, x1: torch.Tensor = None,
                return_features: bool = False):
        """Embeds and projects the input images.

        """

        # forward pass of first input x0
        f0 = self.backbone(x0).flatten(start_dim=1)
        out0 = self.projection_head(f0)

        # append features if requested
        if return_features:
            out0 = (out0, f0)

        # return out0 if x1 is None
        if x1 is None:
            return out0

        # forward pass of second input x1
        f1 = self.backbone(x1).flatten(start_dim=1)
        out1 = self.projection_head(f1)

        # append features if requested
        if return_features:
            out1 = (out1, f1)

        # return both outputs
        return out0, out1