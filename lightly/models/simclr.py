""" ResNet architecture with projection head for implementation of SimCLR. """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn
from lightly.models.resnet import ResNetGenerator


class ResNetSimCLR(nn.Module):
    """ Implementation of ResNet with a projection head.

    Attributes:
        name:
            ResNet version, choose from resnet-{9, 18, 34, 50, 101, 152}.
        width:
            Width of the ResNet.
        num_ftrs:
            Dimension of the embedding (before the projection head).
        out_dim:
            Dimension of the output (after the projection head).

    """

    def __init__(self,
                 name: str ='resnet-18',
                 width: float = 1.,
                 num_ftrs: int = 16,
                 out_dim: int = 128):

        self.num_ftrs = num_ftrs
        self.out_dim = out_dim

        super(ResNetSimCLR, self).__init__()
        resnet = ResNetGenerator(name=name, width=width)

        last_conv_channels = list(resnet.children())[-1].in_features

        self.features = nn.Sequential(
            nn.BatchNorm2d(3),
            *list(resnet.children())[:-1],
            nn.Conv2d(last_conv_channels, num_ftrs, 1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(),
            nn.Linear(num_ftrs, out_dim)
        )

    def forward(self, x: torch.Tensor):
        """ Forward pass through ResNetSimCLR.

        Extracts features with the ResNet backbone and applies the projection
        head to the output space.

        Args:
            x:
                Tensor of shape bsz x channels x W x H

        Returns:
            Tensor of shape bsz x out_dim

        """
        # embed images in feature space
        emb = self.features(x)
        emb = emb.squeeze()

        # return projection to space for loss calcs
        return self.projection_head(emb)
