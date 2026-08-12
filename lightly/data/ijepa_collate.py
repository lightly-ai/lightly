"""Collate function for I-JEPA."""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from __future__ import annotations

import math
from multiprocessing import Value
from multiprocessing.sharedctypes import Synchronized
from typing import Any

import torch
from torch import Tensor


class IJEPAMaskCollator:
    """Collator for IJEPA model [0].

    Experimental: Support for I-JEPA is experimental, there might be breaking changes
    in the future.

    Code inspired by [1].

    - [0]: Joint-Embedding Predictive Architecture, 2023, https://arxiv.org/abs/2301.08243
    - [1]: https://github.com/facebookresearch/ijepa
    """

    def __init__(
        self,
        input_size: int | tuple[int, int] = (224, 224),
        patch_size: int = 16,
        enc_mask_scale: tuple[float, float] = (0.2, 0.8),
        pred_mask_scale: tuple[float, float] = (0.2, 0.8),
        aspect_ratio: tuple[float, float] = (0.3, 3.0),
        nenc: int = 1,
        npred: int = 2,
        min_keep: int = 4,
        allow_overlap: bool = False,
    ) -> None:
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.patch_size = patch_size
        self.height, self.width = (
            input_size[0] // patch_size,
            input_size[1] // patch_size,
        )
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = (
            allow_overlap  # whether to allow overlap b/w enc and pred masks
        )
        # collator is shared across worker processes. Value is typed as returning
        # SynchronizedBase, but the "i" typecode gives a Synchronized[int].
        self._itr_counter: Synchronized[int] = Value("i", -1)  # type: ignore[assignment]

    def step(self) -> int:
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(
        self,
        generator: torch.Generator,
        scale: tuple[float, float],
        aspect_ratio_scale: tuple[float, float],
    ) -> tuple[int, int]:
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(
        self,
        b_size: tuple[int, int],
        acceptable_regions: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        h, w = b_size

        def constrain_mask(mask: Tensor, tries: int = 0) -> None:
            """Helper to restrict given mask to a set of acceptable regions"""
            assert acceptable_regions is not None
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top : top + h, left : left + w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch: list[Any]) -> tuple[Any, Any, Any]:
        """Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        """
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )
        e_size = self._sample_block_size(
            generator=g, scale=self.enc_mask_scale, aspect_ratio_scale=(1.0, 1.0)
        )

        collated_masks_pred: Any = []
        collated_masks_enc: Any = []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):
            masks_p: list[Tensor] = []
            masks_C: list[Tensor] = []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions: list[Tensor] | None = masks_C

            if self.allow_overlap:
                acceptable_regions = None

            masks_e: list[Tensor] = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(
                    e_size, acceptable_regions=acceptable_regions
                )
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [
            [cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred
        ]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [
            [cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc
        ]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred
