""" Momentum Encoder """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import copy

import torch
import torch.nn as nn

@torch.no_grad()
def batch_shuffle(batch: torch.Tensor):
    """Returns the shuffled batch and the indices to undo.

    Examples:
        >>> # forward pass through the momentum model with batch shuffling
        >>> x1_shuffled, shuffle = batch_shuffle(x1)
        >>> f1 = moco_momentum(x1)
        >>> out0 = projection_head_momentum(f0)
        >>> out1 = batch_unshuffle(out1, shuffle)
    """
    batch_size = batch.shape[0]
    shuffle = torch.randperm(batch_size, device=batch.device)
    return batch[shuffle], shuffle

@torch.no_grad()
def batch_unshuffle(batch: torch.Tensor, shuffle: torch.Tensor):
    """Returns the unshuffled batch.

    Examples:
        >>> # forward pass through the momentum model with batch shuffling
        >>> x1_shuffled, shuffle = batch_shuffle(x1)
        >>> f1 = moco_momentum(x1)
        >>> out0 = projection_head_momentum(f0)
        >>> out1 = batch_unshuffle(out1, shuffle)
    """
    unshuffle = torch.argsort(shuffle)
    return batch[unshuffle]

def deactivate_requires_grad(params):
    """Deactivates the requires_grad flag for all parameters.
    
    Examples:
        >>> backbone = resnet18()
        >>> deactivate_requires_grad(backbone)
    """
    for param in params:
        param.requires_grad = False

def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    """Updates parameters of `model_ema` with Exponential Moving Average of `model`

    Examples:
        >>> backbone = resnet18()
        >>> projection_head = MoCoProjectionHead()
        >>> backbone_momentum = copy.deepcopy(moco)
        >>> projection_head_momentum = copy.deepcopy(projection_head)
        >>>
        >>> # update momentum
        >>> update_momentum(moco, moco_momentum, m=0.999)
        >>> update_momentum(projection_head, projection_head_momentum, m=0.999)
    """
    for model_params, model_ema_params in zip(model.parameters(), model_ema.parameters()):
        ema = model_ema_params.data
        curr = model_params.data
        ema = ema * m + (1 - m) * curr

class MomentumWrapper(nn.Module):
    """Module to provide momentum encoder functionalities in one wrapper.

    """
    model: nn.Module
    m: float
    shuffle: bool

    def __init__(self, model: nn.Module, m: float=0.999, shuffle: bool=True):
        self.model_ema = deactivate_requires_grad(copy.deepcopy(model))
        self.m = m
        self.shuffle = shuffle

    def forward(self, x):
        if self.shuffle:
            x, shuffle = batch_shuffle(x)

        out = self.model(x)

        if self.shuffle:
            out = batch_unshuffle(out, shuffle)
        return out

    def update_moving_average(self, model: nn.Module, model_ema: nn.Module):
        update_momentum(model, model_ema, m=self.m)