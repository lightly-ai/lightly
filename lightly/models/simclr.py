""" ResNet architecture with projection head for implementation of SimCLR. """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn

from lightly.models.resnet import ResNetGenerator
from lightly.models.batchnorm import get_norm_layer
from lightly.models._loader import _StateDictLoaderMixin


def _get_features_and_projections(resnet, num_ftrs, out_dim, num_splits):
    """Removes classification head from the ResNet and adds a projection head.

    - Adds a batchnorm layer to the input layer.
    - Replaces the output layer by a Conv2d followed by adaptive average pool.
    - Adds a 2-layer mlp projection head.

    """

    # get the number of features from the last channel
    last_conv_channels = list(resnet.children())[-1].in_features

    # replace output layer
    features = nn.Sequential(
        get_norm_layer(3, num_splits),
        *list(resnet.children())[:-1],
        nn.Conv2d(last_conv_channels, num_ftrs, 1),
        nn.AdaptiveAvgPool2d(1),
    )

    # 2-layer mlp projection head
    projection_head = nn.Sequential(
        nn.Linear(num_ftrs, num_ftrs),
        nn.ReLU(),
        nn.Linear(num_ftrs, out_dim)
    )

    return features, projection_head


class ResNetSimCLR(nn.Module, _StateDictLoaderMixin):
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
                 num_ftrs: int = 32,
                 out_dim: int = 128,
                 num_splits: int = 0):

        self.num_ftrs = num_ftrs
        self.out_dim = out_dim
        self.num_splits = num_splits

        super(ResNetSimCLR, self).__init__()
        resnet = ResNetGenerator(name=name, width=width, num_splits=num_splits)

        self.features, self.projection_head = _get_features_and_projections(
            resnet, self.num_ftrs, self.out_dim, num_splits)

    def load_from_state_dict(self,
                             state_dict,
                             strict: bool = True,
                             apply_filter: bool = True):
        """Initializes a ResNetMoCo and loads weights from a checkpoint.

        Args:
            state_dict:
                State dictionary with layer weights.
            strict:
                Set to False when loading from a partial state_dict.
            apply_filter:
                If True, removes the `model.` prefix from keys in the state_dict.

        """
        self._custom_load_from_state_dict(state_dict, strict, apply_filter)

    def forward(self, x: torch.Tensor):
        """Forward pass through ResNetSimCLR.

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
