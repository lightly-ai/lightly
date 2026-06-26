"""BEiT: BERT Pre-Training of Image Transformers
Paper: https://arxiv.org/abs/2106.08254
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightly.models.modules.masked_image_modeling import BEITEncoder


class BEIT(nn.Module):
    """BEIT masked image modelling (MIM) pre-training model .

    Combines BEITEncoder + MIMHead.  The training loop (masking strategy,
    loss computation, optimiser) lives outside this module – this class
    only defines the forward computation.

    Usage::

        model = BEIT()
        out = model(images, bool_masked_pos)
        # out['mim_logits']  shape (B, N_masked, vocab_size)
        # loss = F.cross_entropy(out['mim_logits'].reshape(-1, V),
        #                        visual_token_ids[bool_masked_pos])
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
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        vocab_size: int = 8192,
    ) -> None:
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
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.mim_head = MIMHead(embed_dim=embed_dim, vocab_size=vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        bool_masked_pos: torch.BoolTensor,
    ) -> dict[str, torch.Tensor]:
        """Args:
            x:               (B, C, H, W)  input images
            bool_masked_pos: (B, N)         True at masked patch positions

        Returns:
              'mim_logits'       – (B, N_masked, vocab_size)  logits for masked positions
              'patch_features'   – (B, N, D)                  all patch encodings
        """
        enc_out = self.encoder(x, bool_masked_pos=bool_masked_pos)
        patch_features = enc_out["patch_features"]

        all_logits = self.mim_head(patch_features)
        mim_logits = all_logits[bool_masked_pos]

        n_masked = bool_masked_pos.sum(dim=1)[0].item()
        B = x.shape[0]
        mim_logits = mim_logits.view(B, int(n_masked), -1)

        return mim_logits, patch_features
