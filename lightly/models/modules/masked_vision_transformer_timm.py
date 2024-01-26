from typing import Optional, Union

import torch
import torch.nn as nn
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, Parameter, Sequential

from lightly.models import utils
from lightly.models.modules.masked_vision_transformer import MaskedVisionTransformer


class MaskedVisionTransformerTIMM(MaskedVisionTransformer):
    def __init__(
        self,
        vit: VisionTransformer,
        mask_token: Union[bool, Parameter],
    ):
        super().__init__(vit=vit, mask_token=mask_token)
        self.vit = vit
        self.mask_token = (
            mask_token
            if isinstance(mask_token, Parameter)
            else Parameter(torch.zeros(1, 1, vit.num_features))
            if mask_token
            else None
        )
        self.sequence_length = vit.patch_embed.num_patches + vit.num_prefix_tokens
        self._initialize_weights()

    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.encode(images, idx_mask=idx_mask, idx_keep=idx_keep)
        if self.vit.attn_pool is not None:
            x = self.vit.attn_pool(x)
        elif self.vit.global_pool == "avg":
            x = x[:, self.vit.num_prefix_tokens :].mean(dim=1)
        elif self.vit.global_pool:
            x = x[:, 0]  # class token
        return x

    def encode(
        self,
        images: torch.Tensor,
        idx_mask: Optional[torch.Tensor] = None,
        idx_keep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # convert images to tokens
        input = self.patch_embed(images)
        # add prefix tokens if needed
        input = self.add_prefix_tokens(input)

        if idx_mask is not None:
            input = utils.mask_at_index(input, idx_mask, self.mask_token)
        # add positional encoding
        input = self.add_pos_embed(input)

        if idx_keep is not None:
            input = utils.get_at_index(input, idx_keep)
        # normalization layer
        input = self.vit.norm_pre(input)
        # apply Transformer blocks
        input = self.vit.blocks(input)
        # normalize
        out: torch.Tensor = self.vit.norm(input)
        return out

    def patch_embed(self, images: Tensor) -> Tensor:
        tokens: Tensor = self.vit.patch_embed(images)
        return tokens

    def add_prefix_tokens(self, x: Tensor) -> Tensor:
        prefix_tokens = []
        if self.vit.cls_token is not None:
            prefix_tokens.append(self.vit.cls_token.expand(x.shape[0], -1, -1))
        if self.vit.reg_token is not None:
            prefix_tokens.append(self.vit.reg_token.expand(x.shape[0], -1, -1))
        if prefix_tokens:
            x = torch.cat(prefix_tokens + [x], dim=1)
        return x

    def add_pos_embed(self, x: Tensor) -> Tensor:
        if self.vit.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.vit.pos_embed,
                (H, W),
                num_prefix_tokens=0
                if self.vit.no_embed_class
                else self.vit.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.vit.pos_embed

        if self.vit.no_embed_class:
            x[:, self.vit.num_prefix_tokens :, :] += pos_embed
        else:
            x = x + pos_embed
        x = self.vit.pos_drop(x)
        return x

    def _initialize_weights(self) -> None:
        # Initialize the patch embedding layer like a linear layer instead of conv
        # layer.
        w = self.vit.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize the class token.
        torch.nn.init.normal_(self.vit.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.vit.apply(_init_weights)

        _initialize_2d_sine_cosine_positional_embedding(self.vit.pos_embed)


def _initialize_2d_sine_cosine_positional_embedding(pos_embedding: Parameter) -> None:
    _, seq_length, hidden_dim = pos_embedding.shape
    grid_size = int((seq_length - 1) ** 0.5)
    sine_cosine_embedding = utils.get_2d_sine_cosine_positional_embedding(
        embed_dim=hidden_dim,
        grid_size=grid_size,
        cls_token=True,
    )
    pos_embedding.data.copy_(
        torch.from_numpy(sine_cosine_embedding).float().unsqueeze(0)
    )
    # Freeze positional embedding.
    pos_embedding.requires_grad = False


def _init_weights(module: Module) -> None:
    if isinstance(module, Linear):
        nn.init.xavier_uniform_(module.weight)
        if isinstance(module, Linear) and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
