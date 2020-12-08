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
    """Implementation of a momentum encoder with a ResNet backbone.

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

    def forward(self, x: torch.Tensor):
        """Embeds and projects the input image.

        Splits the input batch into q and k following the notation of MoCo.
        Extracts features with the ResNet backbone and applies the projection
        head to the output space.

        Args:
            x:
                Tensor of shape bsz x channels x W x H

        Returns:
            Tensor of shape bsz x out_dim

        """
        self._momentum_update(self.m)

        # adopting the notation of the moco paper
        batch_size = x.shape[0] // 2
        q = x[:batch_size]
        k = x[batch_size:]
        
        # embed queries
        emb_q = self.backbone(q).squeeze()
        out_q = self.projection_head(emb_q)

        # embed keys
        with torch.no_grad():

            # shuffle for batchnorm
            if self.batch_shuffle:
                k, shuffle = self._batch_shuffle(k)

            emb_k = self.momentum_backbone(k).squeeze()
            out_k = self.momentum_projection_head(emb_k).detach()
        
            # unshuffle for batchnorm
            if self.batch_shuffle:
                out_k = self._batch_unshuffle(out_k, shuffle)

        return torch.cat([out_q, out_k], axis=0)
