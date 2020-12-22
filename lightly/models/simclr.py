""" SimCLR Model """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn


def _get_simclr_projection_head(num_ftrs: int, out_dim: int):
    """Returns a 2-layer projection head.

    Reference (07.12.2020):
    https://github.com/google-research/simclr/blob/master/model_util.py#L141

    """
    modules = [
        nn.Linear(num_ftrs, num_ftrs),
        #nn.BatchNorm1d(num_ftrs),
        nn.ReLU(),
        nn.Linear(num_ftrs, out_dim)
    ]
    return nn.Sequential(*modules)


class SimCLR(nn.Module):
    """Implementation of the SimCLR architecture.

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
        self.projection_head = _get_simclr_projection_head(num_ftrs, out_dim)

    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None,
                return_features: bool = False):
        """Embeds and projects the input images.

        Extracts features with the backbone and applies the projection
        head to the output space.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H
            x1:
                Tensor of shape bsz x channels x W x H
            return_features:
                TODO

        Returns:
            TODO: Elaborate explanation: Tensor of shape bsz x out_dim

        """
        
        # TODO
        f0 = self.backbone(x0).squeeze()
        out0 = self.projection_head(f0)

        # TODO
        if return_features:
            out0 = (out0, f0)

        # TODO
        if x1 is None:
            return out0

        # TODO
        f1 = self.backbone(x1).squeeze()
        out1 = self.projection_head(f1)

        # TODO
        if return_features:
            out1 = (out1, f1)

        return out0, out1
