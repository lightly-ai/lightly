""" DirectCLR Model """

# Copyright (c) 2025. Lightly AG and its affiliates.
# All Rights Reserved

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class DirectCLR(nn.Module):
    """Implementation of the DirectCLR architecture

    Recommended loss: :py:class:`lightly.loss.ntx_ent_loss.NTXentLoss`

    [0] DirectCLR, 2021, https://arxiv.org/abs/2110.09348

    Attributes:
        backbone:
            Backbone model to extract features from images.
        dim:
            Length of the subvector of the feature vector on which to apply the loss.

    """

    def __init__(self, backbone: nn.Module, dim: int = 32) -> None:
        super(DirectCLR, self).__init__()

        self.backbone = backbone
        self.dim = dim

        warnings.warn(
            Warning(
                "The high-level building block DirectCLR will be deprecated in version 1.3.0. "
                + "Use low-level building blocks instead. "
                + "See https://docs.lightly.ai/self-supervised-learning/lightly.models.html for more information"
            ),
            DeprecationWarning,
        )

    def forward(
        self, x0: Tensor, x1: Optional[Tensor] = None, return_features: bool = False
    ) -> (
        Tensor
        | Tuple[Tensor, Tensor]
        | Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]
    ):
        """Embeds and projects the input images.

        Extracts features with the backbone and takes the first self.dim
        values as subvector of the output space. If both x0 and x1 are not None,
        both will be passed through the backbone and sliced. If x1 is None, only
        x0 will be forwarded.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.
            return_features:
                Whether or not to return the intermediate features backbone(x).

        Returns:
            The subvector output of the embedding of x0 and (if x1 is not None)
            the subvector output of x1. If return_features is True, the output for
            each x is a tuple (out, f) where f are the features before slicing.

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
        f0: Tensor = self.backbone(x0).flatten(start_dim=1)
        out0: Tensor = f0[:, 0 : self.dim]

        # return out0 if x1 is None
        if x1 is None:
            # return features if requested
            if return_features:
                return out0, f0
            return out0

        # forward pass of second input x1
        f1: Tensor = self.backbone(x1).flatten(start_dim=1)
        out1: Tensor = f1[:, 0 : self.dim]

        # return features if requested
        if return_features:
            return (out0, f0), (out1, f1)

        # return both outputs
        return out0, out1
