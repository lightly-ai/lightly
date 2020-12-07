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
    """Implementation of ResNet with a projection head.

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

    def forward(self, x: torch.Tensor):
        """Embeds and projects the input images.

        Extracts features with the backbone and applies the projection
        head to the output space.

        Args:
            x:
                Tensor of shape bsz x channels x W x H

        Returns:
            Tensor of shape bsz x out_dim

        """
        # embed images in feature space
        emb = self.backbone(x).squeeze()

        # return projection to space for loss calcs
        return self.projection_head(emb)
