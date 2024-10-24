from functools import partial
from typing import Callable, List, Union

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from lightly.models import utils


class IJEPAPredictorTIMM(nn.Module):
    """Predictor for the I-JEPA model [0].

    Experimental: Support for I-JEPA is experimental, there might be breaking changes
    in the future.

    Predict patch embeddings. Code inspired by [1].

    - [0]: Joint-Embedding Predictive Architecture, 2023, https://arxiv.org/abs/2301.08243
    - [1]: https://github.com/facebookresearch/ijepa

    Attributes:
        num_patches:
            Number of patches (tokens), including the class token.
        depth:
            Number of transformer blocks.
        mlp_dim:
            Dimension of the MLP in the transformer block.
        predictor_embed_dim:
            Dimension of inner predicted patches(tokens).
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
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        """Initializes the IJEPAPredictorTIMM with the specified dimensions."""

        super().__init__()

        self.predictor_embed = nn.Linear(mlp_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.predictor_proj = nn.Linear(predictor_embed_dim, mlp_dim, bias=True)
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
        )
        predictor_pos_embed = utils.get_2d_sincos_pos_embed(
            self.predictor_pos_embed.shape[-1], int(num_patches**0.5), cls_token=False
        )
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(predictor_pos_embed).float().unsqueeze(0)
        )

        # original implementation also has drop path rate
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=drop_path_rate,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        masks_x: Union[List[torch.Tensor], torch.Tensor],
        masks: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass of the IJEPAPredictorTIMM.

        Args:
            x:
                Input tensor of shape (batch_size, sequence_length, feature_dim).
            masks_x:
                Mask indices for the input tensor of shape (batch_size, num_patches).
            masks:
                Mask indices for the predicted tokens of shape (batch_size, num_patches).

        Returns:
            The predicted output tensor of shape (batch_size, num_patches, output_dim).
        """

        assert (masks is not None) and (
            masks_x is not None
        ), "Cannot run predictor without mask indices"

        len_masks_x = len(masks_x) if isinstance(masks_x, list) else 1
        len_masks = len(masks) if isinstance(masks, list) else 1

        B = len(x) // len_masks_x
        x = self.predictor_embed(x)
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)

        x += self.apply_masks(x_pos_embed, masks_x)
        _, N_ctxt, _ = x.shape

        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = self.apply_masks(pos_embs, masks)
        pos_embs = self.repeat_interleave_batch(pos_embs, B, repeat=len_masks_x)
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)

        pred_tokens += pos_embs
        x = x.repeat(len_masks, 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x

    def repeat_interleave_batch(
        self, x: torch.Tensor, B: int, repeat: int
    ) -> torch.Tensor:
        """Repeat and interleave the input tensor."""
        N = len(x) // B
        x = torch.cat(
            [
                torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0)
                for i in range(N)
            ],
            dim=0,
        )
        return x

    def apply_masks(
        self, x: torch.Tensor, masks: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """Apply masks to the input tensor.

        From https://github.com/facebookresearch/ijepa/blob/main/src/masks/utils.py

        Args:
            x:
                tensor of shape [B (batch-size), N (num-patches), D (feature-dim)].
            masks:
                tensor or list of tensors containing indices of patches in [N] to keep.

        Returns:
            Tensor of shape [B, N', D] where N' is the number of patches to keep.
        """

        if not isinstance(masks, list):
            masks = [masks]

        all_x = []
        for m in masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            all_x += [torch.gather(x, dim=1, index=mask_keep)]
        return torch.cat(all_x, dim=0)
