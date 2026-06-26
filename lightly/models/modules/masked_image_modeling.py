from typing import Optional

import torch
import torch.nn as nn

from lightly.models.utils import _init_weights, drop_path


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and project."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, C, H, W)

        Returns:
            (B, N, D)  where N = num_patches
        """
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj._is_output_proj = True
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class MLP(nn.Module):
    """Feed-forward network inside each Transformer block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2._is_output_proj = True
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    """Single Transformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)
        self.layer_idx = layer_idx

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, (attn_weights if return_attn else None)


class BEITEncoder(nn.Module):
    """BEIT Vision Transformer encoder .

    The [S] (CLS) token is prepended; 1-D learnable position embeddings are added.
    A learnable mask embedding [M] is used to replace masked patches.

    Attributes:
        mask_token  – learnable embedding for masked positions (e[M])
        cls_token   – prepended [S] token
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
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    layer_idx=i + 1,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for i, block in enumerate(self.blocks, start=1):
            for m in block.modules():
                _init_weights(m, layer_idx=i)

    def apply_mask(
        self,
        x: torch.Tensor,
        bool_masked_pos: torch.BoolTensor,
    ) -> torch.Tensor:
        """Replace masked patch positions with the learnable mask embedding.

        Args:
            x:               (B, N, D)  patch embeddings (without CLS)
            bool_masked_pos: (B, N)     True at positions to be masked
        Returns:
            x_masked: (B, N, D)
        """
        mask_tokens = self.mask_token.expand(x.shape[0], x.shape[1], -1)
        x = torch.where(bool_masked_pos.unsqueeze(-1), mask_tokens, x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        return_all_attn: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Args:
            x:               (B, C, H, W)
            bool_masked_pos: (B, N)  True at masked patch positions.
                             If None, no masking is applied (useful for fine-tuning).
            return_all_attn: if True, collect attention maps from every block.

        Returns:
            dict with keys:
              'last_hidden_state'  – (B, N+1, D)
              'patch_features'     – (B, N, D)   patch tokens at last layer
              'cls_feature'        – (B, D)       CLS token at last layer
              'all_attentions'     – list of (B, H, N+1, N+1) if requested, else []
        """
        B = x.shape[0]

        tokens = self.patch_embed(x)

        if bool_masked_pos is not None:
            tokens = self.apply_mask(tokens, bool_masked_pos)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        tokens = tokens + self.pos_embed

        all_attentions = []
        for block in self.blocks:
            tokens, attn = block(tokens, return_attn=return_all_attn)
            if return_all_attn and attn is not None:
                all_attentions.append(attn)

        tokens = self.norm(tokens)

        return {
            "last_hidden_state": tokens,
            "patch_features": tokens[:, 1:],
            "cls_feature": tokens[:, 0],
            "all_attentions": all_attentions,
        }
