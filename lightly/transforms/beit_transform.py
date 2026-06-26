# Copyright (c) 2024. Lightly AG and its affiliates.
# All Rights Reserved

# Implements the data augmentation and blockwise masking strategy described in
# BEIT: BERT Pre-Training of Image Transformers, 2021
# https://arxiv.org/abs/2106.08254

from __future__ import annotations

import math
from typing import Tuple

import torch
import torchvision.transforms as T
from PIL import Image


class BlockwiseMaskingGenerator:
    """Generates random block-shaped masks .

    At each call a random block (with random aspect ratio) is selected and
    added to the mask until the target masking ratio is reached.


    Args:
        num_patches_per_side:
            Grid side length. Default is 14 (for 224×224 images with 16×16
            patches).
        mask_ratio:
            Target fraction of patches to mask (default 0.4).
        min_block_patches:
            Minimum number of patches per block (default 16).
        aspect_ratio_range:
            ``(min_r, max_r)`` range for the block aspect ratio sampled
            log-uniformly. Default is ``(0.3, 1/0.3)``.

    Examples:
        >>> generator = BlockwiseMaskingGenerator(num_patches_per_side=14)
        >>> mask = generator(batch_size=4)  # (4, 196)
    """

    def __init__(
        self,
        num_patches_per_side: int = 14,
        mask_ratio: float = 0.4,
        min_block_patches: int = 16,
        aspect_ratio_range: Tuple[float, float] = (0.3, 1.0 / 0.3),
    ) -> None:
        self.h = num_patches_per_side
        self.w = num_patches_per_side
        self.N = self.h * self.w
        self.target_count = int(math.ceil(mask_ratio * self.N))
        self.min_block = min_block_patches
        self.ar_min, self.ar_max = aspect_ratio_range

    def __call__(self, batch_size: int = 1) -> torch.BoolTensor:
        """Generate a batch of block-shaped boolean masks.

        Args:
            batch_size:
                Number of masks to generate.

        Returns:
            Boolean tensor of shape ``(batch_size, N)`` where ``N =
            num_patches_per_side²``. ``True`` indicates a masked position.
        """
        masks = []
        for _ in range(batch_size):
            mask = torch.zeros(self.h, self.w, dtype=torch.bool)
            while mask.sum().item() < self.target_count:
                remaining = self.target_count - mask.sum().item()
                s = torch.randint(
                    self.min_block,
                    max(self.min_block + 1, int(remaining) + 1),
                    (1,),
                ).item()
                ar = math.exp(
                    torch.empty(1)
                    .uniform_(math.log(self.ar_min), math.log(self.ar_max))
                    .item()
                )
                a = max(1, int(round(math.sqrt(s * ar))))
                b = max(1, int(round(math.sqrt(s / ar))))
                if a > self.h or b > self.w:
                    continue
                t = torch.randint(0, self.h - a + 1, (1,)).item()
                l = torch.randint(0, self.w - b + 1, (1,)).item()
                mask[t : t + a, l : l + b] = True
            masks.append(mask.flatten())
        return torch.stack(masks, dim=0)


class BEITTransform:
    """Image transform for BEIT pre-training.

    Applies the standard BEIT augmentation pipeline (random resized crop,
    horizontal flip, colour jitter, normalisation) and exposes a
    :attr:`mask_generator` for producing blockwise boolean masks.

    In the training loop, call ``transform`` on each PIL image to get the
    augmented tensor, then call ``transform.mask_generator(batch_size=B)``
    once per batch to obtain the mask that is passed to the encoder.


    Attributes:
        mask_generator:
            A :class:`BlockwiseMaskingGenerator` instance configured with the
            parameters supplied to this transform.

    Args:
        input_size:
            Spatial size of the input image after cropping (default 224).
        patch_size:
            Patch size used by the ViT backbone; determines the grid side
            length as ``input_size // patch_size`` (default 16).
        mask_ratio:
            Fraction of patches to mask (default 0.4).
        min_block_patches:
            Minimum number of patches per masked block (default 16).
        color_jitter:
            Colour jitter strength applied to brightness, contrast, saturation
            and hue (default 0.4, matching §2.5 of the paper).
        mean:
            Channel-wise normalisation mean (default ImageNet mean).
        std:
            Channel-wise normalisation std (default ImageNet std).

    Examples:
        >>> from lightly.transforms import BEITTransform
        >>> transform = BEITTransform()
        >>>
        >>> # applied per image in the dataset
        >>> image = transform(pil_image)  # (3, 224, 224)
        >>>
        >>> # called once per batch inside the dataloader collation / training loop
        >>> mask = transform.mask_generator(batch_size=64)  # (64, 196)
    """

    def __init__(
        self,
        input_size: int = 224,
        patch_size: int = 16,
        mask_ratio: float = 0.4,
        min_block_patches: int = 16,
        color_jitter: float = 0.4,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        num_patches_per_side = input_size // patch_size

        self.mask_generator = BlockwiseMaskingGenerator(
            num_patches_per_side=num_patches_per_side,
            mask_ratio=mask_ratio,
            min_block_patches=min_block_patches,
        )

        self.transform = T.Compose(
            [
                T.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),
                T.RandomHorizontalFlip(),
                T.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=color_jitter * 0.25,
                ),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply the augmentation pipeline to a single PIL image.

        Args:
            image:
                Input PIL image.

        Returns:
            Augmented image tensor of shape ``(C, H, W)``.
        """
        return self.transform(image)
