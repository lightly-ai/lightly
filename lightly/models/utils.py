""" Utils for working with SSL models """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

@torch.no_grad()
def batch_shuffle(
    batch: torch.Tensor, 
    distributed: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly shuffles all tensors in the batch.

    Args:
        batch:
            The batch to shuffle.
        distributed:
            If True then batches are shuffles across multiple gpus.

    Returns:
        A (batch, shuffle) tuple where batch is the shuffled version of the 
        input batch and shuffle is an index to restore the original order.

    Examples:
        >>> # forward pass through the momentum model with batch shuffling
        >>> x1_shuffled, shuffle = batch_shuffle(x1)
        >>> f1 = moco_momentum(x1)
        >>> out0 = projection_head_momentum(f0)
        >>> out1 = batch_unshuffle(out1, shuffle)
    """
    if distributed:
        return batch_shuffle_distributed(batch)
    batch_size = batch.shape[0]
    shuffle = torch.randperm(batch_size, device=batch.device)
    return batch[shuffle], shuffle

@torch.no_grad()
def batch_unshuffle(
    batch: torch.Tensor, 
    shuffle: torch.Tensor,
    distributed: bool = False,
) -> torch.Tensor:
    """Unshuffles a batch. 

    Args:
        batch:
            The batch to unshuffle.
        shuffle:
            Index to unshuffle the batch.
        distributed:
            If True then the batch is unshuffled across multiple gpus.
    
    Returns:
        The unshuffled batch.

    Examples:
        >>> # forward pass through the momentum model with batch shuffling
        >>> x1_shuffled, shuffle = batch_shuffle(x1)
        >>> f1 = moco_momentum(x1)
        >>> out0 = projection_head_momentum(f0)
        >>> out1 = batch_unshuffle(out1, shuffle)
    """
    if distributed:
        return batch_unshuffle_distributed(batch, shuffle)
    unshuffle = torch.argsort(shuffle)
    return batch[unshuffle]

@torch.no_grad()
def concat_all_gather(x: torch.Tensor) -> torch.Tensor:
    """Returns concatenated instances of x gathered from all gpus.

    This code was taken and adapted from here:
    https://github.com/facebookresearch/moco.

    """
    output = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(output, x, async_op=False)
    output = torch.cat(output, dim=0)
    return output

@torch.no_grad()
def batch_shuffle_distributed(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shuffles batch over multiple gpus.

    This code was taken and adapted from here:
    https://github.com/facebookresearch/moco.

    Args:
        batch:
            The tensor to shuffle.

    Returns:
        A (batch, shuffle) tuple where batch is the shuffled version of the 
        input batch and shuffle is an index to restore the original order.
    
    """
    # gather from all gpus
    batch_size_this = batch.shape[0]
    batch_gather = concat_all_gather(batch)
    batch_size_all = batch_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    # broadcast to all gpus
    dist.broadcast(idx_shuffle, src=0)

    # index for restoring
    shuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = dist.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return batch_gather[idx_this], shuffle

@torch.no_grad()
def batch_unshuffle_distributed(
    batch: torch.Tensor, 
    shuffle: torch.Tensor
) -> torch.Tensor:
    """Undo batch shuffle over multiple gpus.

    This code was taken and adapted from here:
    https://github.com/facebookresearch/moco.

    Args:
        batch:
            The tensor to unshuffle.
        shuffle:
            Index to restore the original tensor.

    Returns:
        The unshuffled tensor.

    """
    # gather from all gpus
    batch_size_this = batch.shape[0]
    batch_gather = concat_all_gather(batch)
    batch_size_all = batch_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = dist.get_rank()
    idx_this = shuffle.view(num_gpus, -1)[gpu_idx]

    return batch_gather[idx_this]

def deactivate_requires_grad(model: nn.Module):
    """Deactivates the requires_grad flag for all parameters of a model.
    
    This has the same effect as permanently executing the model within a `torch.no_grad()`
    context. Use this method to disable gradient computation and therefore
    training for a model.

    Examples:
        >>> backbone = resnet18()
        >>> deactivate_requires_grad(backbone)
    """
    for param in model.parameters():
        param.requires_grad = False

def activate_requires_grad(model: nn.Module):
    """Activates the requires_grad flag for all parameters of a model.

    Use this method to activate gradients for a model (e.g. after deactivating
    them using `deactivate_requires_grad(...)`).

    Examples:
        >>> backbone = resnet18()
        >>> activate_requires_grad(backbone)
    """
    for param in model.parameters():
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


@torch.no_grad()
def normalize_weight(weight: nn.Parameter, dim: int = 1, keepdim: bool = True):
    """Normalizes the weight to unit length along the specified dimension.

    """
    weight.div_(torch.norm(weight, dim=dim, keepdim=keepdim))
