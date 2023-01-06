""" Utils for working with SSL models """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import math
from typing import Optional, Tuple, Union
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

def repeat_token(token: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Repeats a token size times.

    Args:
        token:
            Token tensor with shape (1, 1, dim).
        size:
            (batch_size, sequence_length) tuple.

    Returns:
        Tensor with shape (batch_size, sequence_length, dim) containing copies
        of the input token.

    """
    batch_size, sequence_length = size
    return token.repeat(batch_size, sequence_length, 1)

def expand_index_like(index: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Expands the index along the last dimension of the input tokens.

    Args:
        index:
            Index tensor with shape (batch_size, idx_length) where each entry is
            an index in [0, sequence_length).
        tokens:
            Tokens tensor with shape (batch_size, sequence_length, dim).

    Returns:
        Index tensor with shape (batch_size, idx_length, dim) where the original
        indices are repeated dim times along the last dimension.

    """
    dim = tokens.shape[-1]
    index = index.unsqueeze(-1).expand(-1, -1, dim)
    return index

def get_at_index(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Selects tokens at index.
    
    Args:
        tokens:
            Token tensor with shape (batch_size, sequence_length, dim).
        index:
            Index tensor with shape (batch_size, index_length) where each entry is
            an index in [0, sequence_length).

    Returns:
        Token tensor with shape (batch_size, index_length, dim) containing the
        selected tokens.

    """
    index = expand_index_like(index, tokens)
    return torch.gather(tokens, 1, index)

def set_at_index(
    tokens: torch.Tensor, 
    index: torch.Tensor, 
    value: torch.Tensor
) -> torch.Tensor:
    """Copies all values into the input tensor at the given indices.
    
    Args:
        tokens:
            Tokens tensor with shape (batch_size, sequence_length, dim).
        index:
            Index tensor with shape (batch_size, index_length).
        value:
            Value tensor with shape (batch_size, index_length, dim).
    
    Returns:
        Tokens tensor with shape (batch_size, sequence_length, dim) containing
        the new values.

    """
    index = expand_index_like(index, tokens)
    return torch.scatter(tokens, 1, index, value)

def mask_at_index(
    tokens: torch.Tensor, 
    index: torch.Tensor, 
    mask_token: torch.Tensor
) -> torch.Tensor:
    """Copies mask token into the input tensor at the given indices.
    
    Args:
        tokens:
            Tokens tensor with shape (batch_size, sequence_length, dim).
        index:
            Index tensor with shape (batch_size, index_length).
        mask_token:
            Value tensor with shape (1, 1, dim).
    
    Returns:
        Tokens tensor with shape (batch_size, sequence_length, dim) containing
        the new values.

    """
    mask = tokens.new_zeros(tokens.shape)
    mask = set_at_index(mask, index, 1)
    return (1 - mask) * tokens + mask * mask_token

def prepend_class_token(
    tokens: torch.Tensor, 
    class_token: torch.Tensor
) -> torch.Tensor:
    """Prepends class token to tokens.
    
    Args:
        tokens:
            Tokens tensor with shape (batch_size, sequence_length, dim).
        class_token:
            Class token with shape (1, 1, dim).
    
    Returns:
        Tokens tensor with the class token prepended at index 0 in every
        sequence. The tensor has shape (batch_size, sequence_length + 1, dim).
    """
    batch_size = tokens.shape[0]
    batch_class_token = class_token.expand(batch_size, -1, -1)
    return torch.cat([batch_class_token, tokens], dim=1)

def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Converts a batch of input images into patches.
    
    Args:
        images:
            Images tensor with shape (batch_size, channels, height, width)
        patch_size:
            Patch size in pixels. Image width and height must be multiples of
            the patch size.

    Returns:
        Patches tensor with shape (batch_size, num_patches, channels * patch_size ** 2)
        where num_patches = image_width / patch_size * image_height / patch_size.

    """
    # N, C, H, W = (batch_size, channels, height, width)
    N, C, H, W = images.shape
    assert H == W and H % patch_size == 0

    patch_h = patch_w = H // patch_size
    num_patches = patch_h * patch_w
    patches = images.reshape(shape=(N, C, patch_h, patch_size, patch_w, patch_size))
    patches = torch.einsum('nchpwq->nhwpqc', patches)
    patches = patches.reshape(shape=(N, num_patches, patch_size ** 2 * C))
    return patches


def random_token_mask(
    size: Tuple[int, int], 
    mask_ratio: float = 0.6,
    mask_class_token: bool = False,
    device: Optional[Union[torch.device, str]] = None,
):
    """Creates random token masks.

    Args:
        size:
            Size of the token batch for which to generate masks.
            Should be (batch_size, sequence_length).
        mask_ratio:
            Percentage of tokens to mask.
        mask_class_token:
            If False the class token is never masked. If True the class token
            might be masked.
        device:
            Device on which to create the index masks.

    Returns:
        A (index_keep, index_mask) tuple where each index is a tensor.
        index_keep contains the indices of the unmasked tokens and has shape
        (batch_size, num_keep). index_mask contains the indices of the masked
        tokens and has shape (batch_size, sequence_length - num_keep). 
        num_keep is equal to sequence_length * (1- mask_ratio).

    """
    batch_size, sequence_length = size
    num_keep = int(sequence_length * (1 - mask_ratio))

    noise = torch.rand(batch_size, sequence_length, device=device)
    if not mask_class_token and sequence_length > 0:
        # make sure that class token is not masked
        noise[:, 0] = -1
        num_keep = max(1, num_keep)

    # get indices of tokens to keep
    indices = torch.argsort(noise, dim=1)
    idx_keep = indices[:, :num_keep]
    idx_mask = indices[:, num_keep:]

    return idx_keep, idx_mask

def _batched_index_select(
    input: torch.Tensor, 
    dim: int, 
    index: torch.Tensor
) -> torch.Tensor:

    """
    Selects elements from the input tensor along a given dimension using the indices in the index tensor.

    Args:
        input: The input tensor.
        dim: The dimension along which to select elements.
        index: A tensor of indices to use for selection.

    Returns:
        A tensor containing the selected elements.

    """

    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def nearest_neighbors(
    input_maps: torch.Tensor,
    candidate_maps: torch.Tensor,
    distances: torch.Tensor, 
    num_matches: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    """Finds the nearest neighbors of the maps in input_maps in candidate_maps.

    Args:
        input_maps: 
            A tensor of maps for which to find nearest neighbors.
            It has size: [batch_size, map_size_0, num_input_maps] 

        candidate_maps: 
            A tensor of maps to search for nearest neighbors.
            It has size: [batch_size, map_size_1, num_candidate_maps]

        distances: 
            A tensor of distances between the maps in input_maps and candidate_maps.
            It has size: [batch_size, map_size_0, map_size_1]

        num_matches: 
            Number of nearest neighbors to return. If num_matches is None or -1,
            all the maps in candidate_maps are considered.

    Returns:
        A tuple of tensors, containing the nearest neighbors in input_maps and candidate_maps.

    """

    if num_matches is None or num_matches == -1:
        num_matches = input_maps.size(1)

    # Find nearest neighbour of each input element in the candidate map 
    topk_values, topk_indices = distances.topk(k=1, dim=2, largest=False) # [bsz, map_size_0, 1]
    topk_values = topk_values.squeeze(-1) # [bsz, map_size_0]
    topk_indices = topk_indices.squeeze(-1) # [bsz, map_size_0]
    
    # Select num_matches neighbors pairs having the lowest distance value.
    _, min_indices = topk_values.topk(k=num_matches, dim=1, largest=False) # [bsz, num_matches]
    
    # Create the filtered input map with num_matches lowest distance values.
    filtered_input_maps = _batched_index_select(input_maps, 1, min_indices) # [bsz, num_matches, num_input_maps]
    
    # Create candidate maps in the same way as input maps, but using corrispondent candidate values
    selected_candidate_maps = torch.gather(candidate_maps, 1, topk_indices.unsqueeze(-1).repeat(1,1, input_maps.shape[2])) # [bsz, map_size_0, num_input_maps]
    filtered_candidate_maps = _batched_index_select(selected_candidate_maps, 1, min_indices) # [bsz, num_matches, num_input_maps]

    return filtered_input_maps, filtered_candidate_maps