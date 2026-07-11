from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from lightly.models import utils
from lightly.models.modules.masked_vision_transformer_decoder_timm import (
    MaskedVisionTransformerDecoderTIMM,
)


class IJEPAPredictorTIMM(MaskedVisionTransformerDecoderTIMM):
    """Predictor for the I-JEPA model [0].

    Experimental: Support for I-JEPA is experimental, there might be breaking changes
    in the future.

    Predict patch embeddings. Code inspired by [1]. Reuses the shared
    :class:`MaskedVisionTransformerDecoderTIMM` building blocks (positional embedding,
    transformer blocks, and norm) and adds the I-JEPA specific input embedding,
    prediction projection, and multi-mask logic.

    - [0]: Joint-Embedding Predictive Architecture, 2023, https://arxiv.org/abs/2301.08243
    - [1]: https://github.com/facebookresearch/ijepa

    Attributes:
        num_patches:
            Number of patches (tokens).
        depth:
            Number of transformer blocks.
        mlp_dim:
            Dimension of the input and output tokens.
        predictor_embed_dim:
            Dimension of inner predicted patches (tokens).
        num_heads:
            Number of attention heads.
        qkv_bias:
            If True, add bias to the query, key, and value tensors.
        mlp_ratio:
            Ratio of mlp hidden dim to embedding dim.
        drop_path_rate:
            Drop paths (Stochastic Depth) per sample.
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
        depth: int,
        mlp_dim: int,
        predictor_embed_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        """Initializes the IJEPAPredictorTIMM with the specified dimensions."""
        super().__init__(
            num_patches=num_patches,
            embed_dim=predictor_embed_dim,
            depth=depth,
            num_heads=num_heads,
            # I-JEPA uses a positional embedding without prefix tokens.
            num_prefix_tokens=0,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            # I-JEPA keeps a zero-initialized mask token and relies on the default
            # timm initialization for the remaining layers. Only the positional
            # embedding is initialized below.
            initialize_weights=False,
        )
        self.predictor_embed = nn.Linear(mlp_dim, predictor_embed_dim, bias=True)
        self.predictor_proj = nn.Linear(predictor_embed_dim, mlp_dim, bias=True)
        utils.initialize_2d_sine_cosine_positional_embedding(
            pos_embedding=self.pos_embed, num_prefix_tokens=0
        )

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        masks_x: list[Tensor] | Tensor,
        masks: list[Tensor] | Tensor,
    ) -> Tensor:
        """Forward pass of the IJEPAPredictorTIMM.

        Args:
            x:
                Input tensor.
            masks_x:
                Mask indices for the input tensor.
            masks:
                Mask indices for the predicted tokens.

        Returns:
            The predicted output tensor.
        """
        assert (masks is not None) and (masks_x is not None), (
            "Cannot run predictor without mask indices"
        )

        len_masks_x = len(masks_x) if isinstance(masks_x, list) else 1
        len_masks = len(masks) if isinstance(masks, list) else 1

        B = len(x) // len_masks_x
        x = self.predictor_embed(x)
        x_pos_embed = self.pos_embed.repeat(B, 1, 1)

        x += utils.apply_masks(x_pos_embed, masks_x)
        _, N_ctxt, _ = x.shape

        pos_embs = self.pos_embed.repeat(B, 1, 1)
        pos_embs = utils.apply_masks(pos_embs, masks)
        pos_embs = utils.repeat_interleave_batch(pos_embs, B, repeat=len_masks_x)
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)

        pred_tokens += pos_embs
        x = x.repeat(len_masks, 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        x = self.decode(x)

        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x
