""" SimSiam Model """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn


def _prediction_mlp(in_dims: int, 
                    h_dims: int, 
                    out_dims: int) -> nn.Sequential:
    """Prediction MLP. The original paper's implementation has 2 layers, with 
    BN applied to its hidden fc layers but no BN or ReLU on the output fc layer.

    Note that the hidden dimensions should be smaller than the input/output 
    dimensions (bottleneck structure). The default implementation using a 
    ResNet50 backbone has an input dimension of 2048, hidden dimension of 512, 
    and output dimension of 2048

    Args:
        in_dims:
            Input dimension of the first linear layer.
        h_dims: 
            Hidden dimension of all the fully connected layers (should be a
            bottleneck!)
        out_dims: 
            Output Dimension of the final linear layer.

    Returns:
        nn.Sequential:
            The projection head.
    """
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Linear(h_dims, out_dims)

    prediction = nn.Sequential(l1, l2)
    return prediction


def _projection_mlp(in_dims: int,
                    h_dims: int,
                    out_dims: int,
                    num_layers: int = 3) -> nn.Sequential:
    """Projection MLP. The original paper's implementation has 3 layers, with 
    BN applied to its hidden fc layers but no ReLU on the output fc layer. 
    The CIFAR-10 study used a MLP with only two layers.

    Args:
        in_dims:
            Input dimension of the first linear layer.
        h_dims: 
            Hidden dimension of all the fully connected layers.
        out_dims: 
            Output Dimension of the final linear layer.
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

    l3 = nn.Sequential(nn.Linear(h_dims, out_dims),
                       nn.BatchNorm1d(out_dims))

    if num_layers == 3:
        projection = nn.Sequential(l1, l2, l3)
    elif num_layers == 2:
        projection = nn.Sequential(l1, l3)
    else:
        raise NotImplementedError("Only MLPs with 2 and 3 layers are implemented.")

    return projection


class SimSiam(nn.Module):
    """ Implementation of SimSiam network

    Attributes:
        backbone:
            Backbone model to extract features from images.
        num_ftrs:
            Dimension of the embedding (before the projection head).
        proj_hidden_dim:
            Dimension of the hidden layer of the projection head. This should
            be the same size as `num_ftrs`.
        pred_hidden_dim:
            Dimension of the hidden layer of the predicion head. This should
            be `num_ftrs` / 4.
        out_dim:
            Dimension of the output (after the projection head).

    """

    def __init__(self,
                 backbone: nn.Module,
                 num_ftrs: int = 2048,
                 proj_hidden_dim: int = 2048,
                 pred_hidden_dim: int = 512,
                 out_dim: int = 2048,
                 num_mlp_layers: int = 3):

        super(SimSiam, self).__init__()

        self.backbone = backbone
        self.num_ftrs = num_ftrs
        self.proj_hidden_dim = proj_hidden_dim
        self.pred_hidden_dim = pred_hidden_dim
        self.out_dim = out_dim

        self.projection_mlp = \
            _projection_mlp(num_ftrs, proj_hidden_dim, out_dim, num_mlp_layers)

        self.prediction_mlp = \
            _prediction_mlp(out_dim, pred_hidden_dim, out_dim)
        
    def forward(self, 
                x0: torch.Tensor, 
                x1: torch.Tensor,
                return_features: bool = False):
        """Forward pass through SimSiam.

        Extracts features with the backbone and applies the projection
        head to the output space. 

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.
            return_features:
                Whether or not to return the intermediate features backbone(x).

        Returns:
            The output projection of x0 and the output projection of x1. If 
            return_features is True, the output for each x is a tuple (out, f) 
            where f are the features before the projection head.

        Args:
            x:
                Tensor of shape bsz x channels x W x H

        Returns:
            Tensor of shape bsz x out_dim

        """
        f0, f1 = self.backbone(x0).squeeze(), self.backbone(x1).squeeze()

        z0, z1 = self.projection_mlp(f0), self.projection_mlp(f1)
        p0, p1 = self.prediction_mlp(z0), self.prediction_mlp(z1)

        out0, out1 = (z0, p0), (z1, p1)

        # append features if requested
        if return_features:
            out0, out1 = (out0, f0), (out1, f1)

        return out0, out1
