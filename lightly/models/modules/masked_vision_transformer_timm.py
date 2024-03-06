import math
from typing import Optional

import torch
import torch.nn as nn
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, Parameter

from lightly.models import utils
from lightly.models.modules.masked_vision_transformer import MaskedVisionTransformer


class MaskedVisionTransformerTIMM(MaskedVisionTransformer, Module):
    """Masked Vision Transformer class using TIMM.

    Attributes:
        vit:
            The VisionTransformer object of TIMM.
        mask_token:
            The mask token.

    """

    def __init__(
        self,
        vit: VisionTransformer,
        mask_token: Optional[Parameter] = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.mask_token = (
            mask_token
            if mask_token is not None
            else Parameter(torch.zeros(1, 1, self.vit.embed_dim))
        )
        self._initialize_weights()

    @property
    def sequence_length(self) -> int:
        seq_len: int = self.vit.patch_embed.num_patches + self.vit.num_prefix_tokens
        return seq_len

    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns encoded class tokens from a batch of images.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_size, image_size).
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                If specified, the indexed tokens are masked with self.mask_token.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                If specified, only the indexed tokens will be passed to the
                encoder.

        Returns:
            Tensor with shape (batch_size, vit.embed_dim) containing the
            encoded class token for every image.

        """
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
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode input images.

        Args:
            input:
                Batch of input images.
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                If specified, the indexed tokens are masked with self.mask_token.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                If specified, only the indexed tokens will be encoded.

        Returns:
            Batch of encoded output tokens.
        """
        # convert images to tokens
        input = self.images_to_tokens(images)
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
        out: Tensor = self.vit.norm(input)
        return out

    def images_to_tokens(self, images: Tensor) -> Tensor:
        """Converts images into patch tokens.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_size, image_size).

        Returns:
            Tensor with shape (batch_size, vit.patch_embed.num_patches, vit.embed_dim)
            containing the patch tokens (excluding prefix tokens).
        """
        tokens: Tensor = self.vit.patch_embed(images)
        if self.vit.dynamic_img_size:
            tokens = tokens.permute(0, 3, 1, 2)  # NHWC -> NCHW
            tokens = tokens.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return tokens

    def add_prefix_tokens(self, x: Tensor) -> Tensor:
        """Adds prefix tokens to image patch tokens.

        Args:
            x:
                Tensor with shape (batch_size, vit.patch_embed.num_patches, vit.embed_dim)
                containing the image patch tokens

        Returns:
            Tensor with shape (batch_size, self.sequence_length, vit.embed_dim) containing
            the image patch tokens and prefix tokens.
        """
        prefix_tokens = []
        if self.vit.cls_token is not None:
            prefix_tokens.append(self.vit.cls_token.expand(x.shape[0], -1, -1))
        if self.vit.reg_token is not None:
            prefix_tokens.append(self.vit.reg_token.expand(x.shape[0], -1, -1))
        if prefix_tokens:
            x = torch.cat(prefix_tokens + [x], dim=1)
        return x

    def add_pos_embed(self, x: Tensor) -> Tensor:
        """Adds positional embeddings to the input tensor based on the Vision Transformer
        (ViT) architecture in vit.

        Args:
            x:
                Input tensor with shape (batch_size, self.sequence_length, vit.embed_dim).

        Returns:
            Tensor after adding positional embeddings, with the same shape as the input.
        """

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
        torch.nn.init.normal_(self.vit.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(_init_weights)

        utils.initialize_2d_sine_cosine_positional_embedding(
            pos_embedding=self.vit.pos_embed, has_class_token=self.vit.has_class_token
        )


def _init_weights(module: Module) -> None:
    if isinstance(module, Linear):
        nn.init.xavier_uniform_(module.weight)
        if isinstance(module, Linear) and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
