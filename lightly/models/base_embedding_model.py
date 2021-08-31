""" Base Embedding Model """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
from typing import List, Tuple

import torch
import torch.nn as nn

from lightly.models.modules.heads import ProjectionHead


class BaseEmbeddingModel(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 head: ProjectionHead
                 ):
        super(BaseEmbeddingModel, self).__init__()
        self.backbone = backbone
        self.head = head

    def _base_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass throught the backbone and head"""
        features = self.backbone(x).flatten(start_dim=1)
        out = self.head(features)
        return out, features


    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None,
                return_features: bool = False):
        """Forward pass through the embedding mdoel.

        Extracts features with the backbone and applies the projection
        head and prediction head(s) to the output space. If both x0 and x1 are not
        None, both will be passed through the backbone, projection, and
        prediction head. If x1 is None, only x0 will be forwarded.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.
            return_features:
                Whether or not to return the intermediate features backbone(x).

        Returns:
            The output prediction and projection of x0 and (if x1 is not None)
            the output prediction and projection of x1. If return_features is
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

        # forward pass of first input x0
        out0, f0 = self._base_forward(x0)

        # return out0 if x1 is None
        if x1 is None:
            # append features if requested
            if return_features:
                return (out0, f0)
            else:
                return out0

        # forward pass of second input x1
        out1, f1 = self._base_forward(x1)

        # append features if requested
        if return_features:
            return (out0, f0), (out1, f1)
        else:
            return out0, out1