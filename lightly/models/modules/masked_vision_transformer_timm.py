import math
import sys
from typing import Callable, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

try:
    from timm.layers import LayerType, Mlp, PatchEmbed
    from timm.layers.pos_embed import resample_abs_pos_embed
    from timm.models import vision_transformer
except ImportError:
    print(
        "TIMM is not available. Please install if you would like to use the TIMM Masked Vision Transformer."
    )
    sys.exit(1)

from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, Parameter

from lightly.models import utils
from lightly.models.modules.masked_vision_transformer import MaskedVisionTransformer


class MaskedVisionTransformerTIMM(
    MaskedVisionTransformer, vision_transformer.VisionTransformer  # type: ignore
):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 32,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        embed_layer: Callable[..., PatchEmbed] = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = vision_transformer.Block,
        mlp_layer: Type[nn.Module] = Mlp,
        mask_token: Optional[Parameter] = None,
    ) -> None:
        MaskedVisionTransformer.__init__(self, mask_token=mask_token)
        vision_transformer.VisionTransformer.__init__(
            self,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            global_pool=global_pool,
            in_chans=in_chans,
            depth=depth,
            num_heads=num_heads,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            init_values=init_values,
            class_token=class_token,
            no_embed_class=no_embed_class,
            reg_tokens=reg_tokens,
            pre_norm=pre_norm,
            fc_norm=fc_norm,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            drop_rate=drop_rate,
            pos_drop_rate=pos_drop_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
            mlp_layer=mlp_layer,
        )
        self.sequence_length = self.patch_embed.num_patches + self.num_prefix_tokens
        self._initialize_weights()

    @classmethod  # type: ignore
    def from_vit(
        cls,
        vit: vision_transformer.VisionTransformer,
        initialize_weights: bool = True,
        mask_token: Optional[Parameter] = None,
    ):
        """Creates a Backbone from a timm ViT model.

        Args:
            vit:
                A timm ViT model.
            initialize_weights:
                If True, then all weights are initialized as in MAE paper. Set this to
                False if vit is pretrained.
            mask_token: The mask token to be used.

        Returns:
            A Backbone with the same architecture as vit.

        """

        backbone = cls(
            img_size=vit.patch_embed.img_size,
            patch_size=vit.patch_embed.patch_size,
            num_classes=vit.num_classes,
            global_pool=vit.global_pool,
            class_token=vit.has_class_token,
            reg_tokens=vit.num_reg_tokens,
            no_embed_class=vit.no_embed_class,
            embed_dim=vit.embed_dim,
            dynamic_img_size=vit.dynamic_img_size,
        )
        backbone.num_prefix_tokens = vit.num_prefix_tokens
        backbone.cls_token = vit.cls_token
        backbone.reg_token = vit.reg_token
        backbone.pos_embed = vit.pos_embed
        backbone.pos_drop = vit.pos_drop
        backbone.patch_drop = vit.patch_drop
        backbone.norm_pre = vit.norm_pre
        backbone.patch_embed = vit.patch_embed
        backbone.blocks = vit.blocks
        backbone.norm = vit.norm
        backbone.attn_pool = vit.attn_pool
        backbone.fc_norm = vit.fc_norm
        backbone.head_drop = vit.head_drop
        backbone.head = vit.head
        backbone.mask_token = mask_token

        if initialize_weights:
            backbone._initialize_weights()
        return backbone

    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.encode(images, idx_mask=idx_mask, idx_keep=idx_keep)
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        return x

    def encode(
        self,
        images: torch.Tensor,
        idx_mask: Optional[torch.Tensor] = None,
        idx_keep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        input = self.norm_pre(input)
        # apply Transformer blocks
        input = self.blocks(input)
        # normalize
        out: torch.Tensor = self.norm(input)
        return out

    def images_to_tokens(self, images: Tensor) -> Tensor:
        tokens: Tensor = self.patch_embed(images)
        if self.dynamic_img_size:
            tokens = tokens.permute(0, 3, 1, 2)  # NHWC -> NCHW
            tokens = tokens.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return tokens

    def add_prefix_tokens(self, x: Tensor, prepend_class_token: bool = True) -> Tensor:
        prefix_tokens = []
        if self.cls_token is not None:
            prefix_tokens.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            prefix_tokens.append(self.reg_token.expand(x.shape[0], -1, -1))
        if prefix_tokens:
            x = torch.cat(prefix_tokens + [x], dim=1)
        return x

    def add_pos_embed(self, x: Tensor) -> Tensor:
        x_prefix = (
            x[:, : self.num_prefix_tokens, :]
            if self.num_prefix_tokens
            else torch.empty()
        )
        x = x[:, self.num_prefix_tokens :, :] if self.num_prefix_tokens else x
        if self.dynamic_img_size:
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
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        if self.no_embed_class:
            x = x + pos_embed
            if self.num_prefix_tokens:
                out = torch.cat((x_prefix, x), dim=1)
        else:
            if self.num_prefix_tokens:
                out = torch.cat((x_prefix, x), dim=1)
            out = out + pos_embed
        out = self.pos_drop(out)
        return out

    def _initialize_weights(self) -> None:
        # Initialize the patch embedding layer like a linear layer instead of conv
        # layer.
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize the class token.
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(_init_weights)

        utils.initialize_2d_sine_cosine_positional_embedding(self.pos_embed)


def _init_weights(module: Module) -> None:
    if isinstance(module, Linear):
        nn.init.xavier_uniform_(module.weight)
        if isinstance(module, Linear) and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
