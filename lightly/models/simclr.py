""" ResNet architecture with projection head for implementation of SimCLR. """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn

from lightly.models.resnet import ResNetGenerator
from lightly.models._helpers import filter_state_dict


def _get_features_and_projections(resnet, num_ftrs, out_dim):
    """Removes classification head from the ResNet and adds a projection head.

    - Adds a batchnorm layer to the input layer.
    - Replaces the output layer by a Conv2d followed by adaptive average pool.
    - Adds a 2-layer mlp projection head.

    """

    # get the number of features from the last channel
    last_conv_channels = list(resnet.children())[-1].in_features

    # replace output layer
    features = nn.Sequential(
        nn.BatchNorm2d(3),
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
                 num_ftrs: int = 32,
                 out_dim: int = 128):

        self.num_ftrs = num_ftrs
        self.out_dim = out_dim

        super(ResNetSimCLR, self).__init__()
        resnet = ResNetGenerator(name=name, width=width)

        self.features, self.projection_head = _get_features_and_projections(
            resnet, self.num_ftrs, self.out_dim)

    @classmethod
    def from_state_dict(cls,
                        state_dict: dict,
                        name: str = 'resnet-18',
                        width: float = 1.,
                        num_ftrs: int = 32,
                        out_dim: int = 128,
                        strict: bool = True,
                        apply_filter: bool = True):
        """Initializes a ResNetMoCo and loads weights from a checkpoint.

        Args:
            state_dict:
                State dictionary with layer weights.
            name:
                ResNet version, choose from resnet-{9, 18, 34, 50, 101, 152}.
            width:
                Width of the ResNet.
            num_ftrs:
                Dimension of the embedding (before the projection head).
            out_dim:
                Dimension of the output (after the projection head).
            strict:
                Set to False when loading from a partial state_dict.
            apply_filter:
                If True, removes the `model.` prefix from keys in the state_dict.

        """
        model = cls(
            name=name,
            width=width,
            num_ftrs=num_ftrs,
            out_dim=out_dim,
        )

        # remove the model. prefix which is caused by the pytorch-lightning
        # checkpoint saver and load the model from the "filtered" state dict
        # this approach is compatible with pytorch_lightning 0.7.1 - 0.8.4 (latest)
        if apply_filter:
            state_dict_ = filter_state_dict(state_dict)
        else:
            state_dict_ = state_dict

        model.load_state_dict(state_dict_, strict=strict)

        return model

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
