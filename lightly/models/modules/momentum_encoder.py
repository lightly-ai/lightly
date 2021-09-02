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


class MomentumEncoder(nn.Module):
    """Mixin to provide momentum encoder functionalities.

    Provides the following functionalities:
        - Momentum encoder initialization.
        - Momentum updates.
        - Batch shuffling and unshuffling.

    """

    def __init__(self, backbone, projection_head, momentum: float = 0.999,
                 batch_shuffle: bool = True):
        """Initializes momentum backbone and a momentum projection head.

        """
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head
        self.momentum = momentum
        self.batch_shuffle = batch_shuffle

        self.momentum_backbone = copy.deepcopy(backbone)
        self.momentum_projection_head = copy.deepcopy(projection_head)
        _deactivate_requires_grad(self.momentum_backbone.parameters())
        _deactivate_requires_grad(self.momentum_projection_head.parameters())

    def forward(self, x):
        self._momentum_update(self.momentum)
        
        # shuffle for batchnorm
        if self.batch_shuffle:
            x, shuffle = self._batch_shuffle(x)

        # run x through momentum encoder
        features = self.momentum_backbone(x).flatten(start_dim=1)
        out = self.momentum_projection_head(features)

        # unshuffle for batchnorm
        if self.batch_shuffle:
            features = self._batch_unshuffle(features, shuffle)
            out = self._batch_unshuffle(out, shuffle)

        return out

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