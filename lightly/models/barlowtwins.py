""" Barlow Twins resnet-based Model [0]
[0] Zbontar,J. et.al. 2021. Barlow Twins... https://arxiv.org/abs/2103.03230
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn
from . import ResNetGenerator
# from . since it is imported in __init__ : '.'=lightly.models.resnet

def _projection_head_barlow(in_dims: int,
                    h_dims: int = 8192,
                    out_dims: int = 8192,
                    num_layers: int = 3) -> nn.Sequential:
    """
    Projection MLP. The original paper's implementation [0] has 3 layers, with
    8192 output units each layer. BN and ReLU applied to first and second layer.

    Args:
        in_dims:
            Input dimension of the first linear layer.
        h_dims:
            Hidden dimension of all the fully connected layers.
            8192 on [0].
        out_dims:
            Output Dimension of the final linear layer.
            Dimension of the latent space. 8192 on [0].
        num_layers:
            Controls the number of layers; must be 2 or 3. Defaults to 3.

    Returns:
        nn.Sequential:
            The projection head.
    """
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Sequential(nn.Linear(h_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l3 = nn.Sequential(nn.Linear(h_dims, out_dims))
    #SimSiam and BarlowTwins only differs in one BN layer

    if num_layers == 3:
        projection = nn.Sequential(l1, l2, l3)
    elif num_layers == 2:
        projection = nn.Sequential(l1, l3)
    else:
        raise NotImplementedError("Only MLPs with 2 and 3 layers are implemented.")

    return projection

class BarlowTwins(nn.Module):
    """Implementation of BarlowTwins[0] network.

    Recommended loss: :py:class:`lightly.loss.barlow_twins_loss.BarlowTwinsLoss`

    Default params are the ones explained in the original paper [0].
    [0] Zbontar,J. et.al. 2021. Barlow Twins... https://arxiv.org/abs/2103.03230

    Attributes:
        backbone:
            Backbone model to extract features from images.
            ResNet-50 in original paper [0].
        num_ftrs:
            Dimension of the embedding (before the projection head).
        proj_hidden_dim:
            Dimension of the hidden layer of the projection head. This should
            be the same size as `num_ftrs`.
        out_dim:
            Dimension of the output (after the projection head).

    """

    def __init__(self,
                 backbone: nn.Module,
                 num_ftrs: int = 2048,
                 proj_hidden_dim: int = 8192,
                 out_dim: int = 8192,
                 num_mlp_layers: int = 3):

        super(BarlowTwins, self).__init__()

        self.backbone = backbone
        self.num_ftrs = num_ftrs
        self.proj_hidden_dim = proj_hidden_dim
        self.out_dim = out_dim

        self.projection_mlp = \
            _projection_head_barlow(num_ftrs, proj_hidden_dim, out_dim, num_mlp_layers)

    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None,
                return_features: bool = False):

        """Forward pass through BarlowTwins.

        Extracts features with the backbone and applies the projection
        head to the output space. If both x0 and x1 are not None, both will be
        passed through the backbone and projection. If x1 is None, only x0 will
        be forwarded.
        Barlow Twins only implement a projection head unlike SimSiam.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.
            return_features:
                Whether or not to return the intermediate features backbone(x).

        Returns:
            The output projection of x0 and (if x1 is not None)
            the output projection of x1. If return_features is
            True, the output for each x is a tuple (out, f) where f are the
            features before the projection head.

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
        # forward pass first input
        f0 = self.backbone(x0).squeeze()
        out0 = self.projection_mlp(f0)

        # append features if requested
        if return_features:
            out0 = (out0, f0)

        if x1 is None:
            return out0

        # forward pass second input
        f1 = self.backbone(x1).squeeze()
        out1 = self.projection_mlp(f1)

        # append features if requested
        if return_features:
            out1 = (out1, f1)

        return out0, out1
