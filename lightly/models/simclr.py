""" SimCLR Model """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import warnings

import torch
import torch.nn as nn

from lightly.models.modules import SimCLRProjectionHead


class SimCLR(nn.Module):
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

        super(SimCLR, self).__init__()

        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(num_ftrs, num_ftrs, out_dim)

        warnings.warn(Warning(
            'The high-level building block SimCLR will be deprecated in version 1.2.0. '
            + 'Use low-level building blocks instead. '
            + 'See https://docs.lightly.ai/lightly.models.html for more information'),
            PendingDeprecationWarning)

    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None,
                return_features: bool = False):
        """Embeds and projects the input images.

        Extracts features with the backbone and applies the projection
        head to the output space. If both x0 and x1 are not None, both will be
        passed through the backbone and projection head. If x1 is None, only
        x0 will be forwarded.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.
            return_features:
                Whether or not to return the intermediate features backbone(x).

        Returns:
            The output projection of x0 and (if x1 is not None) the output
            projection of x1. If return_features is True, the output for each x
            is a tuple (out, f) where f are the features before the projection
            head.

        Examples:
            >>> # single input, single output
            >>> out = model(x) 
            >>> 
            >>> # single input with return_features=True
            >>> out, f = model(x, return_features=True)
            >>>
            >>> # two inputs, two outputs
            >>> out0, out1 = model(x0, x1)
            >>>
            >>> # two inputs, two outputs with return_features=True
            >>> (out0, f0), (out1, f1) = model(x0, x1, return_features=True)

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
