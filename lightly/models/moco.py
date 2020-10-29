""" ResNet architecture with projection head and momentum encoder. """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn
from lightly.models.resnet import ResNetGenerator


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


class ResNetMoCo(nn.Module):
    """ Implementation of a momentum encoder with a ResNet backbone.

    Attributes:
        name:
            ResNet version, choose from resnet-{9, 18, 34, 50, 101, 152}.
        width:
            Width of the ResNet.
        num_ftrs:
            Dimension of the embedding (before the projection head).
        out_dim:
            Dimension of the output (after the projection head).
        m:
            Momentum for momentum update of the key-encoder.

    """

    def __init__(self,
                 name: str ='resnet-18',
                 width: float = 1.,
                 num_ftrs: int = 32,
                 out_dim: int = 128,
                 m: float = 0.999):

        self.num_ftrs = num_ftrs
        self.out_dim = out_dim
        self.m = m

        super(ResNetMoCo, self).__init__()

        self.features, self.projection_head = \
            _get_features_and_projections(
                ResNetGenerator(name=name, width=width), self.num_ftrs, self.out_dim)

        self.key_features, self.key_projection_head = \
            _get_features_and_projections(
                ResNetGenerator(name=name, width=width), self.num_ftrs, self.out_dim)

        # set key-encoder weights to query-encoder weights
        for param_k in self.key_features.parameters():
            param_k.requires_grad = False
        for param_k in self.key_projection_head.parameters():
            param_k.requires_grad = False
        self._momentum_update_key_encoder(0.)


    @torch.no_grad()
    def _momentum_update_key_encoder(self, m=0.):
        """Momentum update of the key-encoder

        Args:
            m:
                Floating point in [0, 1] which determines update rate.

        """
        # zip feature weights
        feature_weight_generator = zip(
            self.features.parameters(),
            self.key_features.parameters())
        # zip projection head weights
        head_weight_generator = zip(
            self.projection_head.parameters(),
            self.key_projection_head.parameters())
        # update weights
        for param_q, param_k in feature_weight_generator:
            param_k.data = param_k.data * m + param_q.data * (1. - m)
        for param_q, param_k in head_weight_generator:
            param_k.data = param_k.data * m + param_q.data * (1. - m)


    def forward(self, x: torch.Tensor):
        """Forward pass through ResNetMoCo

        Splits the input batch into q and k following the notation of MoCo.
        Extracts features with the ResNet backbone and applies the projection
        head to the output space.

        Args:
            x:
                Tensor of shape bsz x channels x W x H

        Returns:
            Tensor of shape bsz x out_dim

        """
        self._momentum_update_key_encoder(self.m)

        # adopting the notation of the moco paper
        batch_size = x.shape[0] // 2
        q = x[:batch_size]
        k = x[batch_size:]
        
        # embed queries
        emb_q = self.features(q).squeeze()
        out_q = self.projection_head(emb_q)

        # embed keys
        with torch.no_grad():
            emb_k = self.key_features(k).squeeze()
            out_k = self.key_projection_head(emb_k).detach()

        return torch.cat([out_q, out_k], axis=0)
