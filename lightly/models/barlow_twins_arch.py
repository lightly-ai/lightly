""" Barlow Twins resnet-based Model [0]
[0] Zbontar,J. et.al. 2021. Barlow Twins... https://arxiv.org/abs/2103.03230
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn
from . import ResNetGenerator, SimSiam
# from . since it is imported in __init__ : '.'=lightly.models.resnet

def _projection_head_barlow(in_dims: int,
                    h_dims: int = 8192,
                    out_dims: int = 8192,
                    num_layers: int = 3) -> nn.Sequential:
    """
    Projection MLP. The original paper's implementation [0] has 3 layers, with
    8192 output units each layer. BN and ReLU applied to first and second layer.
    The CIFAR-10 study used a MLP with only two layers.

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

    if num_layers == 3:
        projection = nn.Sequential(l1, l2, l3)
    elif num_layers == 2:
        projection = nn.Sequential(l1, l3)
    else:
        raise NotImplementedError("Only MLPs with 2 and 3 layers are implemented.")

    return projection

class BarlowTwins(nn.Module):

    def __init__(self,
                 backbone: nn.Module = ResNetGenerator('resnet-50'),
                 num_ftrs: int = 2048,
                 proj_hidden_dim: int = 8192,
                 pred_hidden_dim: int = 512, ##OJO
                 out_dim: int = 8192,
                 num_mlp_layers: int = 3):

        super(BarlowTwins, self).__init__()

        bonenet = backbone
        self.backbone = nn.Sequential(
            *list(bonenet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # Barlow Twins uses the same architecture as SimSiam
        # The difference is in the parameters used in [0]
        self.resnet_simsiam = SimSiam(self.backbone,
                                      num_ftrs=num_ftrs,
                                      proj_hidden_dim=proj_hidden_dim,
                                      pred_hidden_dim=pred_hidden_dim, ##OJO
                                      out_dim=out_dim,
                                      num_mlp_layers=num_mlp_layers
                                     )

        def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None,
                return_features: bool = False):

            self.resnet_simsiam(x0, x1, return_features)
