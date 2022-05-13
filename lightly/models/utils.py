""" Utils for working with SSL models """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import math
from typing import Tuple
import warnings

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


# copy paste from PyTorch master branch as it is not available in older releases
# source: https://github.com/pytorch/pytorch/blob/20ac7362009dd8e0aca6e72fc9357773136a83b8/torch/nn/init.py#L22-L54
def _no_grad_trunc_normal(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    a: float,
    b: float,
) -> torch.Tensor:
    """Initializes the input tensor with a truncated normal distribution.

    This method is based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

    Args:
        tensor:
            The tensor to initialize.
        mean:
            Mean of the distribution.
        std:
            Standard deviation of the distribution.
        a:
            Minimum value of the distribution, values below will be clamped.
        b:
            Maximum value of the distribution, values above will be clamped.

    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def repeat_token(token, size):
    # repeats token to have
    batch_size, sequence_length = size[:2]
    return token.repeat(batch_size, sequence_length, 1)

def expand_index_like(idx, input):
    # expands the index along the feature dimension of input
    # returns idx with shape (batch_size, sequence_length, dim_input)
    dim = input.shape[-1]
    idx = idx.unsqueeze(-1).expand(-1, -1, dim)
    return idx

def get_at_index(input, idx):
    # gets tokens at index
    idx = expand_index_like(idx, input)
    return torch.gather(input, 1, idx)

def set_at_index(input, idx, value):
    # sets tokens at index to value
    idx = expand_index_like(idx, input)
    return torch.scatter(input, 1, idx, value)

def prepend_class_token(input, class_token):
    # prepends class token to input
    batch_size = input.shape[0]
    batch_class_token = class_token.expand(batch_size, -1, -1)
    return torch.cat([batch_class_token, input], dim=1)

def patchify(imgs, patch_size):
    # converts images into patches
    # output has shape (N, num_patches, patch_size ** 2 * C)
    N, C, H, W = imgs.shape
    assert H == W and H % patch_size == 0

    patch_h = patch_w = H // patch_size
    num_patches = patch_h * patch_w
    patches = imgs.reshape(shape=(N, C, patch_h, patch_size, patch_w, patch_size))
    patches = torch.einsum('nchpwq->nhwpqc', patches)
    patches = patches.reshape(shape=(N, num_patches, patch_size ** 2 * C))
    return patches


def random_token_mask(
    size, 
    mask_ratio=0.6,
    mask_class_token=False,
    device=None
):
    # creates random masks 
    # returns idx_keep, idx_mask tuple
    # idx_keep has shape (batch_size, num_keep)
    # idx_mask has shape (batch_size, sequence_length - num_keep)
    batch_size, sequence_length = size
    num_keep = int(sequence_length * (1 - mask_ratio))
    
    noise = torch.rand(batch_size, sequence_length, device=device)
    if not mask_class_token:
        # make sure that class token is not masked
        noise[:, 0] = -1
    
    # get indices of tokens to keep
    indices = torch.argsort(noise, dim=1)
    idx_keep = indices[:, :num_keep]
    idx_mask = indices[:, num_keep:]
    
    return idx_keep, idx_mask
