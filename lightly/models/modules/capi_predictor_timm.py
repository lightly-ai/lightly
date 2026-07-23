"""CAPI cross-attention predictor.

- [0]: CAPI: Cluster and Predict Latent Patches for Improved Masked Image Modeling, 2025, https://arxiv.org/abs/2502.08769
- [1]: https://github.com/facebookresearch/capi
- [2]: RoFormer: Rotary Position Embedding, 2021, https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import torch
from timm.layers import RotaryEmbeddingCat, apply_rot_embed_cat
from torch import Tensor
from torch.nn import GELU, LayerNorm, Linear, Module, ModuleList, Parameter, Sequential


class _RoPECrossAttentionBlock(Module):
    """Cross-attention block with rotary position embeddings on the queries and keys."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query_norm = LayerNorm(embed_dim)
        self.context_norm = LayerNorm(embed_dim)
        self.to_query = Linear(embed_dim, embed_dim)
        self.to_key_value = Linear(embed_dim, 2 * embed_dim)
        self.projection = Linear(embed_dim, embed_dim)
        self.mlp_norm = LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Sequential(
            Linear(embed_dim, hidden_dim), GELU(), Linear(hidden_dim, embed_dim)
        )

    def forward(
        self, queries: Tensor, context: Tensor, query_rope: Tensor, context_rope: Tensor
    ) -> Tensor:
        # queries: (batch_size, num_queries, embed_dim)
        # context: (batch_size, num_context, embed_dim)
        # query_rope: (batch_size, num_queries, 2 * head_dim)
        # context_rope: (batch_size, num_context, 2 * head_dim)
        batch_size, num_queries, embed_dim = queries.shape
        num_context = context.shape[1]

        q = self.to_query(self.query_norm(queries))
        q = q.reshape(batch_size, num_queries, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        key_value = self.to_key_value(self.context_norm(context))
        key_value = key_value.reshape(
            batch_size, num_context, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        keys, values = key_value[0], key_value[1]

        # Rotate queries and keys by their grid positions. Unsqueeze to broadcast the
        # rotary embedding over the attention heads.
        q = apply_rot_embed_cat(q, query_rope.unsqueeze(1))
        keys = apply_rot_embed_cat(keys, context_rope.unsqueeze(1))

        attention = torch.softmax(
            (q @ keys.transpose(-2, -1)) * self.head_dim**-0.5, dim=-1
        )
        out = (
            (attention @ values)
            .transpose(1, 2)
            .reshape(batch_size, num_queries, embed_dim)
        )
        queries = queries + self.projection(out)
        queries = queries + self.mlp(self.mlp_norm(queries))
        return queries


class CAPIPredictorTIMM(Module):
    """Cross-attention predictor for CAPI [0] with rotary position embeddings [2].

    A learned mask token is placed at every masked patch position and cross-attends
    to the visible encoder tokens to predict the masked patches. Rotary position
    embeddings encode the 2D grid positions of the query (masked) and context
    (visible) tokens, so the predictor operates on token subsets without a separate
    positional embedding table. This mirrors the reference implementation [1] and
    differs from a masked-autoencoder decoder, which re-inserts mask tokens into the
    full sequence and self-attends over everything.

    This module requires TIMM to be installed.

    - [0]: CAPI: Cluster and Predict Latent Patches for Improved Masked Image Modeling, 2025, https://arxiv.org/abs/2502.08769
    - [1]: https://github.com/facebookresearch/capi
    - [2]: RoFormer: Rotary Position Embedding, 2021, https://arxiv.org/abs/2104.09864

    Attributes:
        embed_dim:
            Dimension of the input and output tokens.
        grid_size:
            Size of the patch grid as (height, width), or a single int for a square
            grid. Used to build the rotary position embeddings.
        depth:
            Number of cross-attention blocks.
        num_heads:
            Number of attention heads.
        mlp_ratio:
            Ratio of the hidden dimension of the MLP to the embedding dimension.

    Raises:
        ValueError: If embed_dim is not divisible by num_heads.
    """

    def __init__(
        self,
        embed_dim: int,
        grid_size: int | tuple[int, int],
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        head_dim = embed_dim // num_heads
        feat_shape = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
        self.rope = RotaryEmbeddingCat(
            dim=head_dim, in_pixels=False, feat_shape=list(feat_shape)
        )
        self.mask_token = Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = ModuleList(
            [
                _RoPECrossAttentionBlock(
                    embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(depth)
            ]
        )
        self.norm = LayerNorm(embed_dim)
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self, context: Tensor, context_positions: Tensor, query_positions: Tensor
    ) -> Tensor:
        """Predicts the masked patches from the visible tokens.

        Args:
            context:
                Visible encoder tokens with shape
                (batch_size, num_context, embed_dim).
            context_positions:
                Row-major grid indices of the visible tokens with shape
                (batch_size, num_context).
            query_positions:
                Row-major grid indices of the masked tokens to predict with shape
                (batch_size, num_queries).

        Returns:
            The predicted tokens for the masked positions with shape
            (batch_size, num_queries, embed_dim).
        """
        rope = self.rope.get_embed().to(context.device)  # (num_patches, 2 * head_dim)
        context_rope = rope[context_positions]
        query_rope = rope[query_positions]
        queries = self.mask_token.expand(context.shape[0], query_positions.shape[1], -1)
        for block in self.blocks:
            queries = block(
                queries=queries,
                context=context,
                query_rope=query_rope,
                context_rope=context_rope,
            )
        predicted: Tensor = self.norm(queries)
        return predicted
