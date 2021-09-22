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
    
    This has the same effect as executing the model within a `torch.no_grad()`
    context. Use this method to disabled gradient computation and therefore
    training for a model.

    Examples:
        >>> backbone = resnet18()
        >>> deactivate_requires_grad(backbone)
    """
    for param in params:
        param.requires_grad = False

def activate_requires_grad(params):
    """Activates the requires_grad flag for all parameters.

    Use this method to activate gradients for a model (e.g. after deactivating
    them using `deactivate_requires_grad(...)`).

    Examples:
        >>> backbone = resnet18()
        >>> activate_requires_grad(backbone)
    """
    for param in params:
        param.requires_grad = True

def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    """Updates parameters of `model_ema` with Exponential Moving Average of `model`

    Momentum encoders are a crucial component fo models such as MoCo or BYOL. 

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
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1. - m)
