""" BYOL Model """

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

import warnings

import torch
import torch.nn as nn

from lightly.models.modules import BYOLProjectionHead
from lightly.models._momentum import _MomentumEncoderMixin


def _get_byol_mlp(num_ftrs: int, hidden_dim: int, out_dim: int):
    """Returns a 2-layer MLP with batch norm on the hidden layer.

    Reference (12.03.2021)
    https://arxiv.org/abs/2006.07733

    """
    modules = [
        nn.Linear(num_ftrs, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim)
    ]
    return nn.Sequential(*modules)


class BYOL(nn.Module, _MomentumEncoderMixin):
    """Implementation of the BYOL architecture.

    Attributes:
        backbone:
            Backbone model to extract features from images.
        num_ftrs:
            Dimension of the embedding (before the projection mlp).
        hidden_dim:
            Dimension of the hidden layer in the projection and prediction mlp.
        out_dim:
            Dimension of the output (after the projection/prediction mlp).
        m:
            Momentum for the momentum update of encoder.
    """

    def __init__(self,
                 backbone: nn.Module,
                 num_ftrs: int = 2048,
                 hidden_dim: int = 4096,
                 out_dim: int = 256,
                 m: float = 0.9):

        super(BYOL, self).__init__()

        self.backbone = backbone
        # the architecture of the projection and prediction head is the same
        self.projection_head = BYOLProjectionHead(num_ftrs, hidden_dim, out_dim)
        self.prediction_head = BYOLProjectionHead(out_dim, hidden_dim, out_dim)
        self.momentum_backbone = None
        self.momentum_projection_head = None

        self._init_momentum_encoder()
        self.m = m

        warnings.warn(Warning(
            'The high-level building block BYOL will be deprecated in version 1.3.0. '
            + 'Use low-level building blocks instead. '
            + 'See https://docs.lightly.ai/lightly.models.html for more information'),
            PendingDeprecationWarning)

    def _forward(self,
                 x0: torch.Tensor,
                 x1: torch.Tensor = None):
        """Forward pass through the encoder and the momentum encoder.

        Performs the momentum update, extracts features with the backbone and
        applies the projection (and prediciton) head to the output space. If
        x1 is None, only x0 will be processed otherwise, x0 is processed with
        the encoder and x1 with the momentum encoder.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.

        Returns:
            The output proejction of x0 and (if x1 is not None) the output 
            projection of x1.
        
        Examples:
            >>> # single input, single output
            >>> out = model._forward(x)
            >>>
            >>> # two inputs, two outputs
            >>> out0, out1 = model._forward(x0, x1)

        """

        self._momentum_update(self.m)

        # forward pass of first input x0
        f0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.projection_head(f0)
        out0 = self.prediction_head(z0)

        if x1 is None:
            return out0

        # forward pass of second input x1
        with torch.no_grad():

            f1 = self.momentum_backbone(x1).flatten(start_dim=1)
            out1 = self.momentum_projection_head(f1)
        
        return out0, out1

    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor,
                return_features: bool = False):
        """Symmetrizes the forward pass (see _forward).

        Performs two forward passes, once where x0 is passed through the encoder
        and x1 through the momentum encoder and once the other way around.

        Note that this model currently requires two inputs for the forward pass 
        (x0 and x1) which correspond to the two augmentations.
        Furthermore, `the return_features` argument does not work yet.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.

        Returns: 
            A tuple out0, out1, where out0 and out1 are tuples containing the
            predictions and projections of x0 and x1: out0 = (z0, p0) and
            out1 = (z1, p1).

        Examples:
            >>> # initialize the model and the loss function
            >>> model = BYOL()
            >>> criterion = SymNegCosineSimilarityLoss()
            >>>
            >>> # forward pass for two batches of transformed images x1 and x2
            >>> out0, out1 = model(x0, x1)
            >>> loss = criterion(out0, out1)

        """

        if x0 is None:
            raise ValueError('x0 must not be None!')
        if x1 is None:
            raise ValueError('x1 must not be None!')

        if not all([s0 == s1 for s0, s1 in zip(x0.shape, x1.shape)]):
            raise ValueError(
                f'x0 and x1 must have same shape but got shapes {x0.shape} and {x1.shape}!'
            )

        p0, z1 = self._forward(x0, x1)
        p1, z0 = self._forward(x1, x0)

        return (z0, p0), (z1, p1)
