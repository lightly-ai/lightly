""" Utils for working with SSL models """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
from __future__ import annotations

import math
import random
import warnings
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn import Identity, Module, Sequential, functional, init
from torch.nn.modules import CrossMapLRN2d, GroupNorm, LayerNorm, LocalResponseNorm
from torch.nn.modules.batchnorm import _NormBase
from torch.nn.parameter import Parameter
from torchvision.ops import StochasticDepth

from lightly.utils import dependency

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from timm.models.vision_transformer import VisionTransformer


def pool_masked(
    source: Tensor, mask: Tensor, num_cls: int, reduce: str = "mean"
) -> Tensor:
    """Reduce image feature maps :math:`(B, C, H, W)` or :math:`(C, H, W)` according to
    an integer index given by `mask` :math:`(B, H, W)` or :math:`(H, W)`.

    Args:
        source: Float tensor of shape :math:`(B, C, H, W)` or :math:`(C, H, W)` to be
            reduced.
        mask: Integer tensor of shape :math:`(B, H, W)` or :math:`(H, W)` containing the
            integer indices.
        num_cls: The number of classes in the possible masks.

    Returns:
        A tensor of shape :math:`(B, C, num_cls)` or :math:`(C, num_cls)`.
    """
    if source.dim() == 3:
        return _mask_reduce(source, mask, reduce, num_cls)
    elif source.dim() == 4:
        return _mask_reduce_batched(source, mask, num_cls)
    else:
        raise ValueError("source must have 3 or 4 dimensions")


def _mask_reduce(
    source: Tensor, mask: Tensor, num_cls: int, reduce: str = "mean"
) -> Tensor:
    output = _mask_reduce_batched(
        source.unsqueeze(0), mask.unsqueeze(0), num_cls=num_cls, reduce=reduce
    )
    return output.squeeze(0)


def _mask_reduce_batched(
    source: Tensor, mask: Tensor, num_cls: int, reduce: str = "mean"
) -> Tensor:
    b, c, h, w = source.shape
    cls = torch.arange(num_cls, device=mask.device)
    num_cls = cls.size(0)
    # create output tensor
    output = source.new_zeros((b, c, num_cls))  # (B C N)
    mask = mask.unsqueeze(1).expand(-1, c, -1, -1).view(b, c, -1)  # (B C HW)
    source = source.view(b, c, -1)  # (B C HW)
    output.scatter_reduce_(
        dim=2, index=mask, src=source, reduce=reduce, include_self=False
    )  # (B C N)
    # scatter_reduce_ produces NaNs if the count is zero
    output = torch.nan_to_num(output, nan=0.0)
    return output


@torch.no_grad()
def batch_shuffle(
    batch: torch.Tensor, distributed: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly shuffles all tensors in the batch.

    Args:
        batch:
            The batch to shuffle.
        distributed:
            If True then batches are shuffled across multiple gpus.

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
    """Shuffles batch over multiple devices.

    This code was taken and adapted from here:
    https://github.com/facebookresearch/moco.

    Args:
        batch:
            The tensor to shuffle.

    Returns:
        A (batch, shuffle) tuple where batch is the shuffled version of the
        input batch and shuffle is an index to restore the original order.

    """
    # gather from all devices
    batch_size_this = batch.shape[0]
    batch_gather = concat_all_gather(batch)
    batch_size_all = batch_gather.shape[0]

    # Calculate the number of devices
    num_devices = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all, device=batch.device)

    # broadcast to all devices
    dist.broadcast(idx_shuffle, src=0)

    # index for restoring
    shuffle = torch.argsort(idx_shuffle)

    # shuffled index for this device
    rank = dist.get_rank()
    idx_this = idx_shuffle.view(num_devices, -1)[rank]
    return batch_gather[idx_this], shuffle


@torch.no_grad()
def batch_unshuffle_distributed(
    batch: torch.Tensor, shuffle: torch.Tensor
) -> torch.Tensor:
    """Undo batch shuffle over multiple devices.

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
    # gather from all devices
    batch_size_this = batch.shape[0]
    batch_gather = concat_all_gather(batch)
    batch_size_all = batch_gather.shape[0]

    # Calculate the number of devices
    num_devices = batch_size_all // batch_size_this

    # Get the rank of the current device
    rank = dist.get_rank()

    # Index for this device after unshuffling
    idx_this = shuffle.view(num_devices, -1)[rank]

    # Returns the unshuffled batch for this device
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


@torch.no_grad()
def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    """Updates parameters of `model_ema` with Exponential Moving Average of `model`

    Momentum encoders are a crucial component for models such as MoCo or BYOL.

    Args:
        model:
            The current model.
        model_ema:
            The model with exponential moving average (EMA) parameters.
        m:
            The momentum factor, between 0 and 1.

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
        model_ema.data = model_ema.data * m + model.data * (1.0 - m)


@torch.no_grad()
def normalize_weight(weight: nn.Parameter, dim: int = 1, keepdim: bool = True):
    """Normalizes the weight to unit length along the specified dimension."""
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
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
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
        tensor.mul_(std * math.sqrt(2.0))
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
    # Expand the index tensor to match the shape of tokens tensor
    index = expand_index_like(index, tokens)

    return torch.gather(tokens, 1, index)


def set_at_index(
    tokens: torch.Tensor, index: torch.Tensor, value: torch.Tensor
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
    tokens: torch.Tensor, index: torch.Tensor, mask_token: torch.Tensor
) -> torch.Tensor:
    """Returns a tensor where the tokens at the given indices are replaced by the
    mask token.

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


def mask_bool(tokens: Tensor, mask: Tensor, mask_token: Tensor) -> Tensor:
    """Returns a tensor with tokens replaced by the mask tokens in all positions where
    the mask is True.

    Args:
        tokens:
            Tokens tensor with shape (batch_size, sequence_length, dim).
        mask:
            Boolean mask tensor with shape (batch_size, sequence_length).
        mask_token:
            Mask token with shape (1, 1, dim).

    Returns:
        Tokens tensor with shape (batch_size, sequence_length, dim) where tokens[i, j]
        is replaced by the mask token if mask[i, j] is True.
    """
    # Convert to int for multiplication.
    mask = mask.unsqueeze(-1).to(torch.bool).to(torch.int)
    return (1 - mask) * tokens + mask * mask_token


def prepend_class_token(
    tokens: torch.Tensor, class_token: torch.Tensor
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

    # Reshape images to form patches
    patches = images.reshape(shape=(N, C, patch_h, patch_size, patch_w, patch_size))

    # Reorder dimensions for patches
    patches = torch.einsum("nchpwq->nhwpqc", patches)

    # Flatten patches
    patches = patches.reshape(shape=(N, num_patches, patch_size**2 * C))

    return patches


def unpatchify(
    patches: torch.Tensor, patch_size: int, channels: int = 3
) -> torch.Tensor:
    """
    Reconstructs images from their patches.

     Args:
         patches:
             Patches tensor with shape (batch_size, num_patches, channels * patch_size ** 2).
         patch_size:
             The patch size in pixels used to create the patches.
         channels:
             The number of channels the image must have

     Returns:
         Reconstructed images tensor with shape (batch_size, channels, height, width).
    """
    N, C = patches.shape[0], channels
    patch_h = patch_w = int(patches.shape[1] ** 0.5)
    assert patch_h * patch_w == patches.shape[1]

    images = patches.reshape(shape=(N, patch_h, patch_w, patch_size, patch_size, C))
    images = torch.einsum("nhwpqc->nchpwq", images)
    images = images.reshape(shape=(N, C, patch_h * patch_size, patch_h * patch_size))
    return images


def random_token_mask(
    size: Tuple[int, int],
    mask_ratio: float = 0.6,
    mask_class_token: bool = False,
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[Tensor, Tensor]:
    """Creates random token masks.

    Args:
        size:
            Size of the token batch for which to generate masks.
            Should be (batch_size, sequence_length).
        mask_ratio:
            Proportion of tokens to mask.
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
        num_keep is equal to sequence_length * (1 - mask_ratio).

    """
    batch_size, sequence_length = size
    # Remove 1 from the considered sequence length if the class token cannot be masked.
    # This only impacts the calculation of the number of tokens to keep.
    mask_sequence_length = sequence_length - int(not mask_class_token)
    num_keep = int(mask_sequence_length * (1 - mask_ratio))

    noise = torch.rand(batch_size, sequence_length, device=device)
    if not mask_class_token and sequence_length > 0:
        # Make sure that class token is not masked
        noise[:, 0] = -1
        num_keep = max(1, num_keep + 1)

    # Get indices of tokens to keep by sorting the noise
    indices = torch.argsort(noise, dim=1)
    idx_keep = indices[:, :num_keep]
    idx_mask = indices[:, num_keep:]

    return idx_keep, idx_mask


def random_prefix_mask(
    size: Tuple[int, int],
    max_prefix_length: int,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """Creates a random prefix mask.

    The mask is created by uniformly sampling a prefix length in [0, max_prefix_length]
    for each sequence in the batch. All tokens with an index greater or equal to
    the prefix length are masked.

    Args:
        size:
            Size of the token batch for which to generate masks.
            Should be (batch_size, sequence_length).
        max_prefix_length:
            Maximum length of the prefix to mask.
        device:
            Device on which to create the mask.

    Returns:
        A mask tensor with shape (batch_size, sequence_length) where each entry
        is True if the token should be masked and False otherwise.

    """
    batch_size, sequence_length = size

    # Create an arange tensor and expand it to match batch size
    arange = torch.arange(sequence_length, device=device).expand(
        batch_size, sequence_length
    )

    # Generate random indices for the prefix length
    indices = torch.randint(0, max_prefix_length, (batch_size, 1), device=device)

    # Create the mask based on arange and indices
    mask = arange >= indices

    return mask


def random_block_mask(
    size: Tuple[int, int, int],
    batch_mask_ratio: float = 0.5,
    min_image_mask_ratio: float = 0.1,
    max_image_mask_ratio: float = 0.5,
    min_num_masks_per_block: int = 4,
    max_num_masks_per_block: Optional[int] = None,
    min_block_aspect_ratio: float = 0.3,
    max_block_aspect_ratio: Optional[float] = None,
    max_attempts_per_block: int = 10,
    device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    """Creates a random block mask for a batch of images.

    A block is in this context a rectangle of patches in an image that are
    masked together. The function generates block masks until the desired number of
    patches per image are masked. DINOv2 uses a more complex masking strategy that
    only generates masks for mask_ratio of the images. On top of that, it also masks
    a different number of patches for every image. This is controlled by the
    min_image_mask_ratio and max_image_mask_ratio arguments.

    Based on the implementation of the block mask in DINOv2 [0]. For details see [1]
    and [2].

    - [0]: DINOv2, 2023, https://arxiv.org/abs/2304.07193
    - [1]: https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/masking.py
    - [2]: https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/collate.py

    Args:
        size:
            Size of the image batch for which to generate masks.
            Should be (batch_size, height, width).
        batch_mask_ratio:
            Percentage of images per batch for which to generate block masks.
            The remaining images are not masked.
        min_image_mask_ratio:
            Minimum percentage of the image to mask. In practice, fewer than
            min_image_mask_ratio patches of the image can be masked due to additional
            constraints.
        max_image_mask_ratio:
            Maximum percentage of the image to mask.
        min_num_masks_per_block:
            Minimum number of patches to mask per block.
        max_num_masks_per_block:
            Maximum number of patches to mask per block.
        min_block_aspect_ratio:
            Minimum aspect ratio (height/width) of a masked block.
        max_block_aspect_ratio:
            Maximum aspect ratio (height/width) of a masked block.
        max_attempts_per_block:
            Maximum number of attempts to find a valid block mask for an image.
        device:
            Device on which to create the mask.
    Returns:
        A boolean tensor with shape (batch_size, height, width) where each entry
        is True if the patch should be masked and False otherwise.

    Raises:
        ValueError: If 'max_image_mask_ratio' is less than 'min_image_mask_ratio'.
    """

    if max_image_mask_ratio < min_image_mask_ratio:
        raise ValueError(
            "max_image_mask_ratio must be greater or equal to min_image_mask_ratio."
        )

    # B is batch size(number of images), H is height, W is width
    B, H, W = size
    num_images_masked = int(B * batch_mask_ratio)
    probs = torch.linspace(
        min_image_mask_ratio, max_image_mask_ratio, num_images_masked + 1
    ).tolist()
    image_masks = []
    for prob_min, prob_max in zip(probs[:-1], probs[1:]):
        num_mask = int(H * W * random.uniform(prob_min, prob_max))
        image_masks.append(
            random_block_mask_image(
                size=(H, W),
                num_masks=num_mask,
                min_num_masks_per_block=min_num_masks_per_block,
                max_num_masks_per_block=max_num_masks_per_block,
                min_block_aspect_ratio=min_block_aspect_ratio,
                max_block_aspect_ratio=max_block_aspect_ratio,
                max_attempts_per_block=max_attempts_per_block,
                device=device,
            )
        )

    # Add non-masked images to fill the batch
    for _ in range(num_images_masked, B):
        image_masks.append(torch.zeros((H, W), dtype=torch.bool, device=device))

    random.shuffle(image_masks)
    return torch.stack(image_masks)


def random_block_mask_image(
    size: Tuple[int, int],
    num_masks: int,
    min_num_masks_per_block: int = 4,
    max_num_masks_per_block: Optional[int] = None,
    min_block_aspect_ratio: float = 0.3,
    max_block_aspect_ratio: Optional[float] = None,
    max_attempts_per_block: int = 10,
    device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    """Creates a random block mask for a single image.

    Args:
        size:
            Size of the image for which to generate a mask.
            Should be (height, width).
        num_masks:
            Number of patches to mask.
        min_num_masks_per_block:
            Minimum number of patches to mask per block.
        max_num_masks_per_block:
            Maximum number of patches to mask per block.
        min_block_aspect_ratio:
            Minimum aspect ratio (height/width) of a masked block.
        max_block_aspect_ratio:
            Maximum aspect ratio (height/width) of a masked block.
        max_attempts_per_block:
            Maximum number of attempts to find a valid block mask.
        device:
            Device on which to create the mask.
    Returns:
        A boolean tensor with shape (height, width) where each entry is True if the
        patch should be masked and False otherwise.

    Raises:
        ValueError: If 'max_num_masks_per_block' is less than 'min_num_masks_per_block' or
            if 'max_block_aspect_ratio' is less than 'min_block_aspect_ratio'
    """

    if max_block_aspect_ratio is None:
        max_block_aspect_ratio = 1 / min_block_aspect_ratio
    if max_num_masks_per_block is None:
        max_num_masks_per_block = num_masks

    if max_num_masks_per_block < min_num_masks_per_block:
        raise ValueError(
            "max_num_masks_per_block must be greater or equal to min_num_masks_per_block."
        )
    if max_block_aspect_ratio < min_block_aspect_ratio:
        raise ValueError(
            "max_block_aspect_ratio must be greater or equal to min_block_aspect_ratio."
        )

    log_min_aspect = math.log(min_block_aspect_ratio)
    log_max_aspect = math.log(max_block_aspect_ratio)

    H, W = size
    mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    mask_count = 0
    while mask_count < num_masks:
        # Try masking a block
        max_new_masked = min(num_masks - mask_count, max_num_masks_per_block)
        delta = 0
        for _ in range(max_attempts_per_block):
            target_area = random.uniform(min_num_masks_per_block, max_new_masked)
            aspect_ratio = math.exp(random.uniform(log_min_aspect, log_max_aspect))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)
                num_already_masked = mask[top : top + h, left : left + w].sum().item()
                num_new_masked = h * w - num_already_masked
                if 0 < num_new_masked <= max_new_masked:
                    mask[top : top + h, left : left + w] = 1
                    delta += num_new_masked
            if delta > 0:
                break
        if delta == 0:
            break
        else:
            mask_count += delta
    return mask


def nearest_neighbors(
    input_maps: torch.Tensor,
    candidate_maps: torch.Tensor,
    distances: torch.Tensor,
    num_matches: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finds the nearest neighbors of the maps in input_maps in candidate_maps.

    Args:
        input_maps:
            A tensor of maps for which to find nearest neighbors.
            It has shape: [batch_size, input_map_size, feature_dimension]
        candidate_maps:
            A tensor of maps to search for nearest neighbors.
            It has shape: [batch_size, candidate_map_size, feature_dimension]
        distances:
            A tensor of distances between the maps in input_maps and candidate_maps.
            It has shape: [batch_size, input_map_size, candidate_map_size]
        num_matches:
            Number of nearest neighbors to return. If num_matches is None or -1,
            all the maps in candidate_maps are considered.

    Returns:
        A tuple of tensors, containing the nearest neighbors in input_maps and candidate_maps.
        They both have shape: [batch_size, input_map_size, feature_dimension]
    """

    if num_matches is None or num_matches == -1 or num_matches > input_maps.size(1):
        num_matches = input_maps.size(1)

    # Find nearest neighbour of each input element in the candidate map
    topk_values, topk_indices = distances.topk(
        k=1, dim=2, largest=False
    )  # [bsz, input_map_size, 1]
    topk_values = topk_values.squeeze(-1)  # [bsz, input_map_size]

    # Select num_matches neighbors pairs having the lowest distance value.
    _, min_indices = topk_values.topk(
        k=num_matches, dim=1, largest=False
    )  # [bsz, num_matches]

    # Create the filtered input map with num_matches lowest distance values.
    feature_dimension = input_maps.shape[2]
    filtered_input_maps = torch.gather(
        input_maps, 1, min_indices.unsqueeze(-1).expand(-1, -1, feature_dimension)
    )  # [bsz, num_matches, feature_dimension]

    # Create candidate maps in the same way as input maps, but using corresponding candidate values
    selected_candidate_maps = torch.gather(
        candidate_maps, 1, topk_indices.expand(-1, -1, feature_dimension)
    )  # [bsz, input_map_size, feature_dimension]
    filtered_candidate_maps = torch.gather(
        selected_candidate_maps,
        1,
        min_indices.unsqueeze(-1).expand(-1, -1, feature_dimension),
    )  # [bsz, num_matches, feature_dimension]

    return filtered_input_maps, filtered_candidate_maps


def most_similar_index(
    x: Tensor,
    y: Tensor,
) -> Tensor:
    """For each feature in x, searches the most similar feature in y and returns the
    corresponding index.

    Args:
        x:
            Tensor with shape (B, N, C) containing the features to compare.
        y:
            Tensor with shape (B, N, C) containing the features to search for similarity.
    Returns:
        Index with shape (B, N) such that y[i, index[i, j]] is most similar to x[i, j]
        over all y[i, ...].

    """
    # Normalize the input tensors along the last dimension
    x = functional.normalize(x, dim=-1)
    y = functional.normalize(y, dim=-1)

    similarity = torch.einsum("bnc,bmc->bnm", x, y)
    return similarity.argmax(dim=2)


def select_most_similar(
    x: Tensor,
    y: Tensor,
    y_values: Tensor,
) -> Tensor:
    """For each feature in x, searches the most similar feature in y and returns the
    corresponding value from y_values.

    Args:
        x:
            Tensor with shape (B, N, C).
        y:
            Tensor with shape (B, N, C).
        y_values:
            Tensor with shape (B, N, D).
    Returns:
        Values with shape (B, N, D) where values[i, j] is the entry in y_values[i, ...]
        such that x[i, j] is the most similar to y[i, ...].
    """
    y_index = most_similar_index(x, y)
    y_index = y_index.unsqueeze(-1).expand(y_values.shape)
    return torch.gather(y_values, dim=1, index=y_index)


_NORM_LAYERS = (_NormBase, LayerNorm, CrossMapLRN2d, LocalResponseNorm, GroupNorm)


def get_weight_decay_parameters(
    modules: Iterable[Module],
    decay_norm: bool = False,
    decay_bias: bool = False,
    norm_layers: Tuple[Type[Module], ...] = _NORM_LAYERS,
) -> Tuple[List[Parameter], List[Parameter]]:
    """Returns all parameters of the modules that should be decayed and not decayed.

    Args:
        modules:
            List of modules to get the parameters from.
        decay_norm:
            If True, normalization parameters are decayed.
        decay_bias:
            If True, bias parameters are decayed.
        norm_layers:
            Tuple of normalization classes to decay if decay_norm is True.

    Returns:
        (params, params_no_weight_decay) tuple.
    """
    params = []
    params_no_weight_decay = []

    # Iterate through each module and categorize its parameters into ones that should be decayed and those that should not
    for module in modules:
        for mod in module.modules():
            if isinstance(mod, norm_layers):
                if not decay_norm:
                    params_no_weight_decay.extend(mod.parameters(recurse=False))
                else:
                    params.extend(mod.parameters(recurse=False))
            else:
                for name, param in mod.named_parameters(recurse=False):
                    if not decay_bias and name.endswith("bias"):
                        params_no_weight_decay.append(param)
                    else:
                        params.append(param)
    return params, params_no_weight_decay


def get_named_leaf_modules(module: Module) -> Dict[str, Module]:
    """Returns all leaf modules of the model with their names."""
    return {
        name: mod for name, mod in module.named_modules() if not any(mod.children())
    }


def add_stochastic_depth_to_blocks(vit: Module, prob: float = 0.0, mode="row") -> None:
    """Adds stochastic depth dropout to all transformer blocks in a Vision Transformer Model

    Args:
        vit:
            Vision Transformer Model to which stochastic depth dropout will be added.
        prob:
            Probability of dropping a layer.
        mode:
            Mode for stochastic depth. Default is "row".

    Raises:
        Runtime Error: If torchvision version is less than 0.12.
    """
    if dependency.torchvision_vit_available():
        # Requires torchvision >=0.12
        from torchvision.models.vision_transformer import EncoderBlock
    else:
        raise RuntimeError("add_stochastic_depth_to_blocks requires torchvision>=0.12.")

    if prob <= 0:
        return

    for mod in vit.modules():
        if isinstance(mod, EncoderBlock):
            mod.dropout = Sequential(mod.dropout, StochasticDepth(p=prob, mode=mode))
            mod.mlp = Sequential(mod.mlp, StochasticDepth(p=prob, mode=mode))


def initialize_positional_embedding(
    pos_embedding: Parameter,
    strategy: str,
    num_prefix_tokens: int,
) -> None:
    """Initializes the positional embedding with the given strategy.

    Args:
        pos_embedding:
            Positional embedding parameter.
        strategy:
            Positional embedding initialization strategy. Valid options are:
            ['learn', 'sincos', 'skip']. 'learn' makes the embedding learnable,
            'sincos' creates a fixed 2D sine-cosine positional embedding, and 'skip'
            does not initialize the positional embedding.
        num_prefix_tokens:
            Number of prefix tokens in the positional embedding. This includes the class
            token.
    Raises:
        ValueError: If an invalid strategy is provided.
    """
    strategies = ["learn", "sincos", "skip"]

    # Validate the strategy
    if strategy not in strategies:
        raise ValueError(
            f"Invalid positional embedding strategy: '{strategy}'. Valid options are: "
            f"{strategies}."
        )

    # Initialize the positional embedding based on the startegy
    if strategy == "learn":
        initialize_learnable_positional_embedding(pos_embedding)
    elif strategy == "sincos":
        initialize_2d_sine_cosine_positional_embedding(
            pos_embedding=pos_embedding,
            has_class_token=num_prefix_tokens > 0,
        )
    elif strategy == "skip":
        return


def initialize_learnable_positional_embedding(pos_embedding: Parameter) -> None:
    """Initializes a learnable positional embedding.

    Uses standard initialization for ViT models, see [0].

    - [0]: https://github.com/huggingface/pytorch-image-models/blob/cec70b6779ea81cec0ca08ee4a257b52affd235a/timm/models/vision_transformer.py#L590

    Args:
        pos_embedding:
            Positional embedding parameter.
    """
    init.trunc_normal_(pos_embedding, std=0.02)
    pos_embedding.requires_grad = True


def initialize_2d_sine_cosine_positional_embedding(
    pos_embedding: Parameter, has_class_token: bool
) -> None:
    _, seq_length, hidden_dim = pos_embedding.shape
    grid_size = int((seq_length - int(has_class_token)) ** 0.5)
    sine_cosine_embedding = get_2d_sine_cosine_positional_embedding(
        embed_dim=hidden_dim,
        grid_size=grid_size,
        cls_token=has_class_token,
    )
    pos_embedding.data.copy_(
        torch.from_numpy(sine_cosine_embedding).float().unsqueeze(0)
    )
    # Freeze positional embedding.
    pos_embedding.requires_grad = False


def get_2d_sine_cosine_positional_embedding(
    embed_dim: int, grid_size: int, cls_token: bool
) -> NDArray[np.float32]:
    """Generates 2D sine-cosine positional embedding.

    Code follows: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

    Args:
        embed_dim:
            Embedding dimension.
        grid_size:
            Height and width of the grid.
        cls_token:
            If True, a positional embedding for the class token is generated.
    Returns:
        Positional embedding with shape (grid_size * grid_size, embed_dim) or
        (1 + grid_size * grid_size, embed_dim) if cls_token is True.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sine_cosine_positional_embedding_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# TODO(guarin): Remove alias and rename function instead. get_2d_sincos_pos_embed
# was introduced by ijepa while get_2d_sine_cosine_positional_embedding was introduced
# by mae.
get_2d_sincos_pos_embed = get_2d_sine_cosine_positional_embedding


def get_2d_sine_cosine_positional_embedding_from_grid(
    embed_dim: int, grid: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Generates 2D sine-cosine positional embedding from a grid.

    Code follows: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

    Args:
        embed_dim:
            Embedding dimension.
        grid:
            Grid of shape (2, grid_size, grid_size) with x and y coordinates.
    Returns:
        Positional embedding with shape (grid_size * grid_size, embed_dim).
    """
    assert embed_dim % 2 == 0

    # Use half of dimensions to encode grid_h
    # (grid_size * grid_size, embed_dim/2)
    emb_h = get_1d_sine_cosine_positional_embedding_from_positions(
        embed_dim // 2, grid[0]
    )

    # Use the other half of dimensions to encode grid_w
    # (grid_size * grid_size, embed_dim/2)
    emb_w = get_1d_sine_cosine_positional_embedding_from_positions(
        embed_dim // 2, grid[1]
    )

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (grid_size * grid_size, embed_dim)
    return emb


def get_1d_sine_cosine_positional_embedding_from_positions(
    embed_dim: int, pos: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Generates 1D sine-cosine positional embedding from positions.

    Code follows: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

    Args:
        embed_dim:
            Embedding dimension.
        pos:
            Positions to be encoded with shape (N, M).
    Returns:
        Positional embedding with shape (N * M, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (embed_dim/2,)

    pos = pos.reshape(-1)  # (N*M,)
    out = np.einsum("m,d->md", pos, omega)  # (N*M, embed_dim/2), outer product

    emb_sin = np.sin(out)  # (N*M, embed_dim/2)
    emb_cos = np.cos(out)  # (N*M, embed_dim/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (N*M, embed_dim)
    return emb


def normalize_mean_var(x: Tensor, dim: int = -1, eps: float = 1.0e-6) -> Tensor:
    """Normalizes the input tensor to zero mean and unit variance.

    Args:
        x:
            Input tensor.
        dim:
            Dimension along which to compute mean and standard deviation. Takes last
            dimension by default.
        eps:
            Epsilon value to avoid division by zero.

    Returns:
        Normalized tensor.
    """
    mean = x.mean(dim=dim, keepdim=True)
    var = x.var(dim=dim, keepdim=True)
    return (x - mean) / (var + eps).sqrt()


def update_drop_path_rate(
    model: VisionTransformer,
    drop_path_rate: float,
    mode: str = "linear",
) -> None:
    """Updates the drop path rate in a TIMM VisionTransformer model.

    Args:
        model:
            TIMM VisionTransformer model.
        drop_path_rate:
            Maximum drop path rate.
        mode:
            Drop path rate update mode. Can be "linear" or "uniform". Linear increases
            the drop path rate from 0 to drop_path_rate over the depth of the model.
            Uniform sets the drop path rate to drop_path_rate for all blocks.
    Raises:
        ValueError: If an unknown mode is provided.
    """
    from timm.layers import DropPath

    total_depth = len(model.blocks)

    # Determine drop path rates based on the specified mode
    if mode == "linear":
        drop_probabilities = np.linspace(0, drop_path_rate, total_depth)
    elif mode == "uniform":
        drop_probabilities = [drop_path_rate for _ in range(total_depth)]
    else:
        raise ValueError(
            f"Unknown mode: '{mode}', supported modes are 'linear' and 'uniform'."
        )

    # Update the drop path rate for each block in the model
    for block, drop_prob in zip(model.blocks, drop_probabilities):
        if drop_prob > 0.0:
            block.drop_path1 = DropPath(drop_prob=drop_path_rate)
            block.drop_path2 = DropPath(drop_prob=drop_path_rate)
        else:
            block.drop_path1 = Identity()
            block.drop_path2 = Identity()


def repeat_interleave_batch(x: Tensor, B: int, repeat: int) -> Tensor:
    """Repeat and interleave the input tensor.

    Args:
        x:
            Tensor with shape (B * N, ...) where B is the batch size and N the number of
            batches.
        B:
            Batch size.
        repeat:
            Number of times to repeat each batch.

    Returns:
        Tensor with shape (B * repeat * N, ...) where each batch is repeated `repeat`
        times.
    """
    N = len(x) // B
    x = torch.cat(
        [
            torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0)
            for i in range(N)
        ],
        dim=0,
    )
    return x


def apply_masks(x: Tensor, masks: Tensor | list[Tensor]) -> Tensor:
    """Apply masks to the input tensor.

    From https://github.com/facebookresearch/ijepa/blob/main/src/masks/utils.py

    Args:
        x:
            Tensor of shape (B, N, D) where N is the number of patches.
        masks:
            Tensor or list of tensors containing indices of patches in
            [0, N-1] to keep. Each tensor musth have shape (B, K) where K is the number
            of patches to keep. All masks must have the same K.

    Returns:
        Tensor of shape (B * num_masks, K, D) where K is the number of patches to keep.
    """

    if not isinstance(masks, list):
        masks = [masks]

    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)
