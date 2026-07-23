from functools import partial
from typing import Any, Callable, Optional, cast

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from torch import Tensor
from torch.nn import LayerNorm, Parameter, Sequential

from lightly.models import utils
from lightly.models.modules.masked_vision_transformer_decoder import (
    MaskedVisionTransformerDecoder,
)
from lightly.models.modules.masked_vision_transformer_timm import init_weights


class MaskedVisionTransformerDecoderTIMM(MaskedVisionTransformerDecoder):
    """Masked Vision Transformer decoder using TIMM.

    Takes patch tokens as input and forwards them through a stack of transformer
    blocks. It owns the mask token, the fixed 2D sine-cosine positional embedding, the
    transformer blocks, and the final norm layer. The input embedding and prediction
    head are kept outside of the decoder so that it can be reused by different methods
    (see for example :class:`IJEPAPredictorTIMM`). Code inspired by [0].

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377

    Attributes:
        num_patches:
            Number of patches.
        embed_dim:
            Embedding dimension of the decoder.
        depth:
            Number of transformer blocks.
        num_heads:
            Number of attention heads.
        num_prefix_tokens:
            Number of prefix tokens (e.g. class or register tokens) preceding the
            patch tokens. Determines the length of the positional embedding.
        mlp_ratio:
            Ratio of mlp hidden dim to embedding dim.
        qkv_bias:
            If True, add bias to the query, key, and value tensors.
        proj_drop_rate:
            Percentage of elements set to zero after the MLP in the transformer.
        attn_drop_rate:
            Percentage of elements set to zero after the attention head.
        drop_path_rate:
            Drop paths (Stochastic Depth) per sample.
        norm_layer:
            Normalization layer.
        initialize_weights:
            Flag that determines if weights should be initialized.
        mask_token:
            The mask token.

    """

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        num_prefix_tokens: int = 1,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(LayerNorm, eps=1e-6),
        initialize_weights: bool = True,
        mask_token: Optional[Parameter] = None,
    ) -> None:
        """Initializes the MaskedVisionTransformerDecoderTIMM with the given parameters."""
        super().__init__()

        self.num_patches = num_patches
        self.num_prefix_tokens = num_prefix_tokens
        self.mask_token = (
            Parameter(torch.zeros(1, 1, embed_dim))
            if mask_token is None
            else mask_token
        )

        # Fixed sine-cosine positional embedding.
        self.pos_embed = Parameter(
            torch.zeros(1, num_patches + num_prefix_tokens, embed_dim),
            requires_grad=False,
        )

        self.blocks = Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate,
                    # timm's type hints for norm_layer vary between versions.
                    norm_layer=cast(Any, norm_layer),
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        if initialize_weights:
            self._initialize_weights()

    @property
    def sequence_length(self) -> int:
        return self.num_patches + self.num_prefix_tokens

    def forward(
        self,
        x: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.preprocess(x, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask)
        x = self.decode(x)
        return x

    def add_pos_embed(self, x: Tensor) -> Tensor:
        out: Tensor = x + self.pos_embed
        return out

    def decode(self, x: Tensor) -> Tensor:
        out: Tensor = self.blocks(x)
        out = self.norm(out)
        return out

    def _initialize_weights(self) -> None:
        torch.nn.init.normal_(self.mask_token, std=0.02)
        utils.initialize_2d_sine_cosine_positional_embedding(
            pos_embedding=self.pos_embed,
            num_prefix_tokens=self.num_prefix_tokens,
        )
        self.apply(init_weights)
