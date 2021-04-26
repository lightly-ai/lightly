""" Momentum Encoder """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import copy

import torch
import torch.nn as nn


def _deactivate_requires_grad(params):
    """Deactivates the requires_grad flag for all parameters.

    """
    for param in params:
        param.requires_grad = False


def _do_momentum_update(prev_params, params, m):
    """Updates the weights of the previous parameters.

    """
    for prev_param, param in zip(prev_params, params):
        prev_param.data = prev_param.data * m + param.data * (1. - m)


class _MomentumEncoderMixin:
    """Mixin to provide momentum encoder functionalities.

    Provides the following functionalities:
        - Momentum encoder initialization.
        - Momentum updates.
        - Batch shuffling and unshuffling.

    To make use of the mixin, simply inherit from it:

    >>> class MyMoCo(nn.Module, _MomentumEncoderMixin):
    >>>
    >>>     def __init__(self, backbone):
    >>>         super(MyMoCo, self).__init__()
    >>>
    >>>         self.backbone = backbone
    >>>         self.projection_head = get_projection_head()
    >>>
    >>>         # initialize momentum_backbone and momentum_projection_head
    >>>         self._init_momentum_encoder()
    >>>
    >>>     def forward(self, x: torch.Tensor):
    >>>
    >>>         # do the momentum update
    >>>         self._momentum_update(0.999)
    >>>
    >>>         # use momentum backbone
    >>>         y = self.momentum_backbone(x)
    >>>         y = self.momentum_projection_head(y)

    """

    m: float
    backbone: nn.Module
    projection_head: nn.Module
    momentum_backbone: nn.Module
    momentum_projection_head: nn.Module

    def _init_momentum_encoder(self):
        """Initializes momentum backbone and a momentum projection head.

        """
        assert self.backbone is not None
        assert self.projection_head is not None

        self.momentum_backbone = copy.deepcopy(self.backbone)
        self.momentum_projection_head = copy.deepcopy(self.projection_head)
        
        _deactivate_requires_grad(self.momentum_backbone.parameters())
        _deactivate_requires_grad(self.momentum_projection_head.parameters())

    @torch.no_grad()
    def _momentum_update(self, m: float = 0.999):
        """Performs the momentum update for the backbone and projection head.

        """
        _do_momentum_update(
            self.momentum_backbone.parameters(),
            self.backbone.parameters(),
            m=m,
        )
        _do_momentum_update(
            self.momentum_projection_head.parameters(),
            self.projection_head.parameters(),
            m=m,
        )

    @torch.no_grad()
    def _batch_shuffle(self, batch: torch.Tensor):
        """Returns the shuffled batch and the indices to undo.

        """
        batch_size = batch.shape[0]
        shuffle = torch.randperm(batch_size, device=batch.device)
        return batch[shuffle], shuffle

    @torch.no_grad()
    def _batch_unshuffle(self, batch: torch.Tensor, shuffle: torch.Tensor):
        """Returns the unshuffled batch.

        """
        unshuffle = torch.argsort(shuffle)
        return batch[unshuffle]
