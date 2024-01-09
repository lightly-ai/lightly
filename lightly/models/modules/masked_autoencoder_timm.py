from __future__ import annotations

import math
import sys
from functools import partial
from typing import Callable, Literal, Optional, Tuple, Type, Union

try:
    from timm.layers import LayerType, Mlp, PatchEmbed
    from timm.models import vision_transformer
except ImportError:
    print("TIMM is not available. Please install if you would like to use the MAE.")
    sys.exit(1)

import torch
import torch.nn as nn
from torch.nn import LayerNorm, Linear, Module, Parameter, Sequential

from lightly.models import utils


class MAEBackbone(vision_transformer.VisionTransformer):  # type: ignore
    """Backbone for the Masked Autoencoder model [0].

    Converts images into patches and encodes them. Code inspired by [1].
    Note that this implementation uses a fixed positional embedding.

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377
    - [1]: https://github.com/facebookresearch/mae
    - [2]: Early Convolutions Help Transformers See Better, 2021, https://arxiv.org/abs/2106.14881.

    Attributes:
        img_size:
            Input image size.
        patch_size:
            Patch size.
        in_chans:
            Number of image input channels.
        num_classes:
            Number of classes for classification head.
        global_pool:
            Type of global pooling for final sequence (default: 'token').
        embed_dim:
            Transformer embedding dimension.
        depth:
            Depth of transformer.
        num_heads:
            Number of attention heads.
        mlp_ratio:
            Ratio of mlp hidden dim to embedding dim.
        qkv_bias:
            Enable bias for qkv projections if True.
        init_values:
            Layer-scale init values (layer-scale enabled if not None).
        class_token:
            Use class token.
        no_embed_class:
            Don't include position embeddings for class (or reg) tokens.
        reg_tokens:
            Number of register tokens.
        fc_norm:
            Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
        dynamic_img_size:
            If set to True, encoding of variable sized images is handled.
        drop_rate:
            Head dropout rate.
        pos_drop_rate:
            Position embedding dropout rate.
        attn_drop_rate:
            Attention dropout rate.
        drop_path_rate:
            Stochastic depth rate.
        weight_init:
            Weight initialization scheme.
        embed_layer:
            Patch embedding layer.
        norm_layer:
            Normalization layer.
        act_layer:
            MLP activation layer.
        block_fn:
            Transformer block layer.

    """

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
    ) -> None:
        super().__init__(
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
        self._initialize_weights()

    @classmethod  # type: ignore
    def from_vit(
        cls, vit: vision_transformer.VisionTransformer, initialize_weights: bool = True
    ) -> MAEBackbone:
        """Creates a MAEBackbone from a timm ViT model.

        Args:
            vit:
                A timm ViT model.
            initialize_weights:
                If True, then all weights are initialized as in MAE paper. Set this to
                False if vit is pretrained.

        Returns:
            A MAEBackbone with the same architecture as vit.

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

        if initialize_weights:
            backbone._initialize_weights()
        return backbone

    def forward(
        self, images: torch.Tensor, idx_keep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returns encoded class tokens from a batch of images.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_size, image_size).
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                If specified, only the indexed tokens will be passed to the
                encoder.

        Returns:
            Tensor with shape (batch_size, hidden_dim) containing the
            encoded class token for every image.

        """
        out = self.encode(images, idx_keep)
        class_token = out[:, 0]
        return class_token

    def encode(
        self, images: torch.Tensor, idx_keep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returns encoded class and patch tokens from images.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_size, image_size).
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                If specified, only the indexed tokens will be passed to the
                encoder.

        Returns:
            Tensor with shape (batch_size, sequence_length, hidden_dim)
            containing the encoded class and patch tokens for every image.

        """
        # convert images to tokens and add class token if needed
        input = self.images_to_tokens(images, prepend_class_token=True)
        # add positional encoding
        input = input + self.pos_embed
        # get the tokens that are kept
        if idx_keep is not None:
            input = utils.get_at_index(input, idx_keep)
        # apply Transformer blocks
        input = self.blocks(input)
        # normalize
        out: torch.Tensor = self.norm(input)
        return out

    def images_to_tokens(
        self, images: torch.Tensor, prepend_class_token: bool
    ) -> torch.Tensor:
        """Converts images into patch tokens.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_size, image_size).

        Returns:
            Tensor with shape (batch_size, sequence_length - 1, hidden_dim)
            containing the patch tokens.
        """
        tokens: torch.Tensor = self.patch_embed(images)
        if prepend_class_token:
            tokens = utils.prepend_class_token(tokens, self.cls_token)
        return tokens

    def _initialize_weights(self) -> None:
        # Initialize the patch embedding layer like a linear layer instead of conv
        # layer.
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize the class token.
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(_init_weights)

        _initialize_2d_sine_cosine_positional_embedding(self.pos_embed)


class MAEDecoder(Module):
    """Decoder for the Masked Autoencoder model [0].

    Decodes encoded patches and predicts pixel values for every patch.
    Code inspired by [1].

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377
    - [1]: https://github.com/facebookresearch/mae

    Attributes:
        num_patches:
            Number of patches.
        patch_size:
            Patch size.
        in_chans:
            Number of image input channels.
        embed_dim:
            Embedding dimension of the encoder.
        decoder_embed_dim:
            Embedding dimension of the decoder.
        decoder_depth:
            Depth of transformer.
        decoder_num_heads:
            Number of attention heads.
        mlp_ratio:
            Ratio of mlp hidden dim to embedding dim.
        proj_drop_rate:
            Percentage of elements set to zero after the MLP in the transformer.
        attn_drop_rate:
            Percentage of elements set to zero after the attention head.
        norm_layer:
            Normalization layer.

    """

    def __init__(
        self,
        num_patches: int,
        patch_size: int,
        in_chans: int = 3,
        embed_dim: int = 1024,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # positional encoding of the decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = Sequential(
            *[
                vision_transformer.Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch

        self._initialize_weights()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns predicted pixel values from encoded tokens.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, embed_input_dim).

        Returns:
            Tensor with shape (batch_size, seq_length, out_dim).

        """
        out = self.embed(input)
        out = self.decode(out)
        return self.predict(out)

    def embed(self, input: torch.Tensor) -> torch.Tensor:
        """Embeds encoded input tokens into decoder token dimension.

        This is a single linear layer that changes the token dimension from
        embed_input_dim to hidden_dim.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, embed_input_dim)
                containing the encoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, hidden_dim) containing
            the embedded tokens.

        """
        out: torch.Tensor = self.decoder_embed(input)
        return out

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder transformer.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, hidden_dim) containing
                the encoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, hidden_dim) containing
            the decoded tokens.

        """
        output: torch.Tensor = input + self.decoder_pos_embed
        output = self.decoder_blocks(output)
        output = self.decoder_norm(output)
        return output

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        """Predics pixel values from decoded tokens.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, hidden_dim) containing
                the decoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, out_dim) containing
            predictions for each token.

        """
        out: torch.Tensor = self.decoder_pred(input)
        return out

    def _initialize_weights(self) -> None:
        torch.nn.init.normal_(self.mask_token, std=0.02)
        _initialize_2d_sine_cosine_positional_embedding(self.decoder_pos_embed)
        self.apply(_init_weights)


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
