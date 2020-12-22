""" MoCo Model """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn

from lightly.models._momentum import _MomentumEncoderMixin


def _get_moco_projection_head(num_ftrs: int, out_dim: int):
    """Returns a 2-layer projection head.

    Reference (07.12.2020):
    https://github.com/facebookresearch/moco/blob/master/moco/builder.py

    """
    modules = [
        nn.Linear(num_ftrs, num_ftrs),
        nn.ReLU(),
        nn.Linear(num_ftrs, out_dim)
    ]
    return nn.Sequential(*modules)


class MoCo(nn.Module, _MomentumEncoderMixin):
    """Implementation of the MoCo (Momentum Contrast) architecture.

    Attributes:
        backbone:
            Backbone model to extract features from images.
        num_ftrs:
            Dimension of the embedding (before the projection head).
        out_dim:
            Dimension of the output (after the projection head).
        m:
            Momentum for momentum update of the key-encoder.

    """

    def __init__(self,
                 backbone: nn.Module,
                 num_ftrs: int = 32,
                 out_dim: int = 128,
                 m: float = 0.999,
                 batch_shuffle: bool = False):

        super(MoCo, self).__init__()

        self.backbone = backbone
        self.projection_head = _get_moco_projection_head(num_ftrs, out_dim)
        self.momentum_features = None
        self.momentum_projection_head = None

        self.m = m
        self.batch_shuffle = batch_shuffle

        # initialize momentum features and momentum projection head
        self._init_momentum_encoder()

    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None,
                return_features: bool = False):
        """Embeds and projects the input image.

        Splits the input batch into q and k following the notation of MoCo.
        Extracts features with the ResNet backbone and applies the projection
        head to the output space.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H
            x1:
                TODO
            return_features:
                TODO

        Returns:
            Tensor of shape bsz x out_dim

        """
        self._momentum_update(self.m)
        
        # TODO
        f0 = self.backbone(x0).squeeze()
        out0 = self.projection_head(f0)

        # TODO
        if return_features:
            out0 = (out0, f0)

        # TODO
        if x1 is None:
            return out0

        # embed keys
        with torch.no_grad():

            # shuffle for batchnorm
            if self.batch_shuffle:
                x1, shuffle = self._batch_shuffle(x1)

            f1 = self.momentum_backbone(x1).squeeze()
            out1 = self.momentum_projection_head(f1).detach()
        
            # unshuffle for batchnorm
            if self.batch_shuffle:
                f1 = self._batch_unshuffle(f1, shuffle)
                out1 = self._batch_unshuffle(out1, shuffle)

            # TODO
            if return_features:
                out1 = (out1, f1)

        return out0, out1
