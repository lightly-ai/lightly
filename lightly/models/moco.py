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
    """Implementation of the MoCo (Momentum Contrast)[0] architecture.

    Recommended loss: :py:class:`lightly.loss.ntx_ent_loss.NTXentLoss` with 
    a memory bank.

    [0] MoCo, 2020, https://arxiv.org/abs/1911.05722

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

        Performs the momentum update, extracts features with the backbone and 
        applies the projection head to the output space. If both x0 and x1 are
        not None, both will be passed through the backbone and projection head.
        If x1 is None, only x0 will be forwarded.

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
        self._momentum_update(self.m)
        
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
        with torch.no_grad():

            # shuffle for batchnorm
            if self.batch_shuffle:
                x1, shuffle = self._batch_shuffle(x1)

            # run x1 through momentum encoder
            f1 = self.momentum_backbone(x1).flatten(start_dim=1)
            out1 = self.momentum_projection_head(f1).detach()
        
            # unshuffle for batchnorm
            if self.batch_shuffle:
                f1 = self._batch_unshuffle(f1, shuffle)
                out1 = self._batch_unshuffle(out1, shuffle)

            # append features if requested
            if return_features:
                out1 = (out1, f1)

        return out0, out1
