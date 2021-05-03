import torch
import torch.nn as nn

import lightly
from lightly.models._momentum import _MomentumEncoderMixin
from lightly.models.batchnorm import get_norm_layer


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
        self.projection_head = _get_byol_mlp(num_ftrs, hidden_dim, out_dim)
        self.prediction_head = _get_byol_mlp(out_dim, hidden_dim, out_dim)
        self.momentum_backbone = None
        self.momentum_projection_head = None

        self._init_momentum_encoder()
        self.m = m

    def _forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None,
                return_features: bool = False):
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
            return_features:
                Whether or not to return the intermediate features backbone(x).

        Returns:
            The output proejction of x0 and (if x1 is not None) the output 
            projection of x1. If return_features is True, the output for each x 
            is a tuple (out, f) where f are the features before the projection
            head.
        
        Examples:
            >>> # single input, single output
            >>> out = model._forward(x)
            >>>
            >>> # single input with return_features=True
            >>> out, f = model._forward(x, return_features=True)
            >>>
            >>> # two inputs, two outputs
            >>> out0, out1 = model._forward(x0, x1)
            >>>
            >>> # two inputs two outputs with return_features=True
            >>> (out0, f0), (out1, f1) = model._forward(x0, x1, return_features=True)

        """

        self._momentum_update(self.m)

        # forward pass of first input x0
        f0 = self.backbone(x0).squeeze()
        z0 = self.projection_head(f0)
        out0 = self.prediction_head(z0)

        # append features if requested
        if return_features:
            out0 = (out0, f0)

        if x1 is None:
            return out0

        # forward pass of second input x1
        with torch.no_grad():

            f1 = self.momentum_backbone(x1).squeeze()
            out1 = self.momentum_projection_head(f1)
        
            if return_features:
                out1 = (out1, f1)
        
        return out0, out1

    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None,
                return_features: bool = False
                ):
        """Symmetrizes the forward pass (see _forward).

        Performs two forward passes, once where x0 is passed through the encoder
        and x1 through the momentum encoder and once the other way around.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.

        Returns: 
                Tensors after passing through the online and
                target networks.


        """
        p0, z1 = self._forward(x0, x1,return_features=return_features)
        p1, z0 = self._forward(x1, x0,return_features=return_features)

        return (z0, p0), (z1, p1)
