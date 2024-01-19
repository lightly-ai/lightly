from typing import Optional, Union

import torch
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor
from torch.nn import Parameter

from lightly.models import utils
from lightly.models.modules.masked_vision_transformer import MaskedVisionTransformer

import torch
import torch.nn as nn
from torch.nn import LayerNorm, Linear, Module, Parameter, Sequential


class MaskedVisionTransformerTIMM(MaskedVisionTransformer):
    def __init__(
        self,
        vit: VisionTransformer,  # here assume that we always initialize with a vit model from timm
        mask_token: Union[bool, Parameter],
        device : str
    ):
        super().__init__()
        self.vit = vit
        self.mask_token = (
            mask_token
            if isinstance(mask_token, Parameter)
            else Parameter(torch.zeros(1, 1, vit.num_features))
            if mask_token
            else None
        )
        self.sequence_length = vit.patch_embed.num_patches + vit.num_prefix_tokens
        self.device = device

    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
    ) -> Tensor:
        out = self.encode(images, idx_keep, idx_mask)
        class_token = out[:, 0]
        return class_token

    def encode(
        self,
        images: torch.Tensor,
        idx_keep: Optional[torch.Tensor] = None,
        idx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # convert images to tokens
        input: torch.Tensor = self.patch_embed(images)
        # add prefix tokens if needed
        input = self.add_prefix_tokens(input)

        if idx_mask is not None:
            assert self.mask_token is not None
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
        return self.vit.patch_embed(images)

    def add_prefix_tokens(self, x: Tensor) -> Tensor:
        tokens = torch.Tensor([])
        tokens = tokens.to(self.device)
        if self.vit.cls_token is not None:
            tokens = torch.cat((tokens, self.vit.cls_token.expand(x.shape[0], -1, -1)), dim=1)
        if self.vit.reg_token is not None:
            tokens = torch.cat((tokens, self.vit.reg_token.expand(x.shape[0], -1, -1)), dim=1)
        return torch.cat((tokens, x), dim=1)


    def add_pos_embed(self, x: Tensor) -> Tensor:
        if self.vit.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.vit.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.vit.no_embed_class else self.vit.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.vit.pos_embed
        # TODO: give option to either have positional encoding for the prefix tokens or not using the self.vit.no_embed_class. 
        # Here it is assumed that the prefix tokens always have positional encodings.
        x = x + pos_embed
        return self.vit.pos_drop(x)
    
    def _initialize_weights(self) -> None:
        # Initialize the patch embedding layer like a linear layer instead of conv
        # layer.
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize the class token.
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(_init_weights)

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
    
