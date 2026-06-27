"""BEiT: BERT Pre-Training of Image Transformers.

Paper: https://arxiv.org/abs/2106.08254

Typical usage example::

    from lightly.models import beit

    model = beit.BEIT(vocab_size=8192)
    out = model(images, bool_masked_pos=mask)
    # out['mim_logits']  -> (B, N_masked, vocab_size)
    # out['patch_features'] -> (B, N, D)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from lightly.models.modules.heads import MIMHead
from lightly.models.modules.masked_image_modeling import BEITEncoder


class BEIT(nn.Module):
    """BEIT masked image modelling (MIM) pre-training model.

    Combines a BEITEncoder with an MIMHead. The training loop
    (masking strategy, loss computation, optimiser) lives outside
    this module — this class only defines the forward computation.

    Attributes:
        encoder:
            BEIT Vision Transformer encoder.
        mim_head:
            Linear projection head mapping patch features to vocabulary
            logits.

    Usage::

        model = BEIT(vocab_size=8192)
        out = model(images, bool_masked_pos)
        # out['mim_logits']      -> (B, N_masked, vocab_size)
        # out['patch_features']  -> (B, N, D)
        # loss = F.cross_entropy(
        #     out['mim_logits'].reshape(-1, V),
        #     visual_token_ids[bool_masked_pos],
        # )
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        init_values: Optional[float] = None,
        use_abs_pos_emb: bool = True,
        use_rel_pos_bias: bool = False,
        use_shared_rel_pos_bias: bool = False,
        attn_head_dim: Optional[int] = None,
        init_std: float = 0.02,
        vocab_size: int = 8192,
    ) -> None:
        """Initializes BEIT.

        Args:
            img_size:
                Spatial resolution of input images.
            patch_size:
                Size of each patch.
            in_channels:
                Number of input channels.
            embed_dim:
                Dimension of token embeddings.
            depth:
                Number of Transformer blocks.
            num_heads:
                Number of attention heads per block.
            mlp_ratio:
                Ratio of MLP hidden dimension to embedding dimension.
            qkv_bias:
                If True, enable bias in query and value projections.
            qk_scale:
                Override for attention scaling factor.
            drop_rate:
                Dropout rate for MLP and attention output.
            attn_drop_rate:
                Dropout rate for attention weights.
            drop_path_rate:
                Maximum stochastic depth rate.
            init_values:
                If provided and > 0, enables LayerScale with this initial
                value.
            use_abs_pos_emb:
                If True, use learnable absolute position embeddings.
            use_rel_pos_bias:
                If True, use per-block relative position bias.
            use_shared_rel_pos_bias:
                If True, use a single shared relative position bias.
            attn_head_dim:
                If provided, overrides the computed attention head dimension.
            init_std:
                Standard deviation for truncated normal initialization.
            vocab_size:
                Size of the discrete visual vocabulary.
        """
        super().__init__()
        self.encoder = BEITEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            use_abs_pos_emb=use_abs_pos_emb,
            use_rel_pos_bias=use_rel_pos_bias,
            use_shared_rel_pos_bias=use_shared_rel_pos_bias,
            attn_head_dim=attn_head_dim,
            init_std=init_std,
        )
        self.mim_head = MIMHead(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        bool_masked_pos: torch.BoolTensor,
        return_all_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for masked image modelling.

        Args:
            x:
                Input images of shape (B, C, H, W).
            bool_masked_pos:
                Boolean mask of shape (B, N) where True indicates a masked
                patch position.
            return_all_tokens:
                If True, returns logits for all patch positions. If False,
                returns logits only for the masked positions (default).

        Returns:
            Dictionary with keys:
                - 'mim_logits': (B, N_masked, vocab_size) if
                  return_all_tokens is False, else (B, N, vocab_size).
                - 'patch_features': (B, N, D) all patch encodings.
        """
        enc_out = self.encoder(
            x=x,
            bool_masked_pos=bool_masked_pos,
        )
        patch_features = enc_out["patch_features"]

        all_logits = self.mim_head(patch_features=patch_features)

        if return_all_tokens:
            mim_logits = all_logits
        else:
            mim_logits = all_logits[bool_masked_pos]
            n_masked = bool_masked_pos.sum(dim=1)[0].item()
            B = x.shape[0]
            vocab_size = all_logits.shape[-1]
            mim_logits = mim_logits.view(B, int(n_masked), vocab_size)

        return {
            "mim_logits": mim_logits,
            "patch_features": patch_features,
        }
