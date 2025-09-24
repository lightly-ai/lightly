import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, Parameter

from lightly.models import utils
from lightly.models.modules.masked_vision_transformer import MaskedVisionTransformer


class MaskedVisionTransformerTIMM(MaskedVisionTransformer):
    """Masked Vision Transformer class using TIMM.

    Attributes:
        vit:
            The VisionTransformer object of TIMM.
        mask_token:
            The mask token.
        weight_initialization:
            The weight initialization method. Valid options are ['', 'skip']. '' uses
            the default MAE weight initialization and 'skip' skips the weight
            initialization.
        antialias:
            Whether to use antialiasing when resampling the positional embeddings.
        pos_embed_initialization:
            The strategy to initialize the positional embeddings. Valid options are
            ['learn', 'sincos', 'skip'].

    """

    def __init__(
        self,
        vit: VisionTransformer,
        mask_token: Optional[Parameter] = None,
        weight_initialization: str = "",
        antialias: bool = True,
        pos_embed_initialization: str = "sincos",
    ) -> None:
        super().__init__()
        self.vit = vit
        self.mask_token = (
            mask_token
            if mask_token is not None
            else Parameter(torch.zeros(1, 1, self.vit.embed_dim))
        )

        if weight_initialization not in ("", "skip"):
            raise ValueError(
                f"Invalid weight initialization method: '{weight_initialization}'. "
                "Valid options are: ['', 'skip']."
            )
        if weight_initialization != "skip":
            self._initialize_weights()

        utils.initialize_positional_embedding(
            pos_embedding=self.vit.pos_embed,
            strategy=pos_embed_initialization,
            num_prefix_tokens=self.vit.num_prefix_tokens,
        )

        self.antialias = antialias

    @property
    def sequence_length(self) -> int:
        seq_len: int = self.vit.patch_embed.num_patches + self.vit.num_prefix_tokens
        return seq_len

    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.encode(images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask)
        if self.vit.attn_pool is not None:
            x = self.vit.attn_pool(x)
        elif self.vit.global_pool == "avg":
            x = x[:, self.vit.num_prefix_tokens :].mean(dim=1)
        elif self.vit.global_pool:
            x = x[:, 0]  # class token
        return x

    def forward_intermediates(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        norm: bool = False,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        # preprocess images, convert to tokens and add positional embeddings
        tokens = self.preprocess(
            images=images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask
        )
        # normalization layer
        tokens = self.vit.norm_pre(tokens)

        intermediates: List[Tensor] = []
        for blk in self.vit.blocks:
            tokens = blk(tokens)
            intermediates.append(self.vit.norm(tokens) if norm else tokens)

        # normalize
        out: Tensor = self.vit.norm(tokens)

        return out, intermediates

    def encode(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # preprocess images, convert to tokens and add positional embeddings
        tokens: Tensor = self.preprocess(
            images=images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask
        )
        # normalization layer
        tokens = self.vit.norm_pre(tokens)
        # apply Transformer blocks
        tokens = self.vit.blocks(tokens)
        # normalize
        tokens = self.vit.norm(tokens)
        return tokens

    def images_to_tokens(self, images: Tensor) -> Tensor:
        tokens: Tensor = self.vit.patch_embed(images)
        if self.vit.dynamic_img_size:
            tokens = tokens.permute(0, 3, 1, 2)  # NHWC -> NCHW
            tokens = tokens.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return tokens

    def prepend_prefix_tokens(self, x: Tensor) -> Tensor:
        prefix_tokens = []
        if self.vit.cls_token is not None:
            prefix_tokens.append(self.vit.cls_token.expand(x.shape[0], -1, -1))
        if self.vit.reg_token is not None:
            prefix_tokens.append(self.vit.reg_token.expand(x.shape[0], -1, -1))
        if prefix_tokens:
            x = torch.cat(prefix_tokens + [x], dim=1)
        return x

    def add_pos_embed(self, x: Tensor) -> Tensor:
        x_prefix = x[:, : self.vit.num_prefix_tokens, :]
        x = x[:, self.vit.num_prefix_tokens :, :]
        if self.vit.dynamic_img_size:
            x = x.transpose(1, 2)  # NLC -> NCL
            total_size = torch.numel(x)
            batch_size = x.size(0)
            num_channels = x.size(1)
            grid_size = int(math.sqrt(total_size / (batch_size * num_channels)))
            x = x.view(
                x.size(0),
                x.size(1),
                grid_size,
                grid_size,
            )  # NCL -> NCHW

            # NCHW -> NHWC
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.vit.pos_embed,
                (H, W),
                num_prefix_tokens=(
                    0 if self.vit.no_embed_class else self.vit.num_prefix_tokens
                ),
                antialias=self.antialias,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.vit.pos_embed

        if self.vit.no_embed_class:
            x = x + pos_embed
            if self.vit.num_prefix_tokens:
                x = torch.cat((x_prefix, x), dim=1)
        else:
            if self.vit.num_prefix_tokens:
                x = torch.cat((x_prefix, x), dim=1)
            x = x + pos_embed
        out: Tensor = self.vit.pos_drop(x)
        return out

    def _initialize_weights(self) -> None:
        # Initialize the patch embedding layer like a linear layer instead of conv
        # layer.
        w = self.vit.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize the class token.
        if self.vit.has_class_token:
            torch.nn.init.normal_(self.vit.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights)


def init_weights(module: Module) -> None:
    if isinstance(module, Linear):
        nn.init.xavier_uniform_(module.weight)
        if isinstance(module, Linear) and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
