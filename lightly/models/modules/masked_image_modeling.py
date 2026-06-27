# Copyright (c) 2024. Lightly AG and its affiliates.
# All Rights Reserved


from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightly.models.utils import drop_path


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    Attributes:
        drop_prob:
            Probability of dropping a path. If 0, the module is a no-op.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        """Initializes DropPath.

        Args:
            drop_prob:
                Probability of dropping a path. Must be in [0, 1).
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies stochastic depth to the input tensor.

        Args:
            x:
                Input tensor of arbitrary shape.

        Returns:
            The input tensor, possibly with some samples zeroed out.
        """
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """Image to patch embedding.

    Splits an image into non-overlapping patches and projects each patch
    into an embedding vector via a convolution.

    Attributes:
        img_size:
            Spatial resolution of the expected input image as (H, W).
        patch_size:
            Size of each patch as (patch_h, patch_w).
        patch_shape:
            Grid shape (num_patches_h, num_patches_w).
        num_patches:
            Total number of patches, i.e. patch_shape[0] * patch_shape[1].
        proj:
            Convolution layer that performs the patch embedding.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """Initializes PatchEmbed.

        Args:
            img_size:
                Spatial resolution of the input image. Can be a single int
                for square images or a tuple (H, W).
            patch_size:
                Size of each patch. Can be a single int or a tuple.
            in_channels:
                Number of input channels (e.g. 3 for RGB).
            embed_dim:
                Dimension of the output embedding vectors.
        """
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.img_size: Tuple[int, int] = img_size
        self.patch_size: Tuple[int, int] = patch_size
        self.patch_shape: Tuple[int, int] = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches: int = self.patch_shape[0] * self.patch_shape[1]

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects image to patch embeddings.

        Args:
            x:
                Input image tensor of shape (B, C, H, W).

        Returns:
            Patch embeddings of shape (B, N, D) where N is the number of
            patches and D is the embedding dimension.

        Raises:
            AssertionError:
                If the input spatial dimensions do not match img_size.
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model "
            f"({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x).flatten(start_dim=2).transpose(dim0=1, dim1=2)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with optional relative position bias.

    This implementation follows the original BEIT / BERT design where
    query and value have separate bias terms while key has no bias.

    Attributes:
        num_heads:
            Number of attention heads.
        scale:
            Scaling factor for query-key dot products (1/sqrt(head_dim)).
        qkv:
            Linear projection for Q, K, V (without bias).
        q_bias:
            Learnable bias for the query projection. None if qkv_bias=False.
        v_bias:
            Learnable bias for the value projection. None if qkv_bias=False.
        relative_position_bias_table:
            Learnable relative position bias table. None if window_size is None.
        relative_position_index:
            Registered buffer holding the index mapping for relative positions.
        attn_drop:
            Dropout layer applied to attention weights.
        proj:
            Linear projection mapping concatenated heads back to dim.
        proj_drop:
            Dropout layer applied to the output of proj.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        window_size: Optional[Tuple[int, int]] = None,
        attn_head_dim: Optional[int] = None,
    ) -> None:
        """Initializes Attention.

        Args:
            dim:
                Total dimension of the input (and output).
            num_heads:
                Number of attention heads.
            qkv_bias:
                If True, add learnable bias to query and value projections.
            qk_scale:
                Override for the attention scaling factor. If None, uses
                1/sqrt(head_dim).
            attn_drop:
                Dropout rate applied to attention weights.
            proj_drop:
                Dropout rate applied to the output projection.
            window_size:
                If provided, enables per-head relative position bias with
                the given (height, width) patch grid shape.
            attn_head_dim:
                If provided, overrides the computed head dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or (head_dim**-0.5)

        self.qkv = nn.Linear(
            in_features=dim,
            out_features=all_head_dim * 3,
            bias=False,
        )
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size is not None:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1
            ) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads)
            )
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
            coords_flatten = torch.flatten(coords, start_dim=1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1,) * 2,
                dtype=relative_coords.dtype,
            )
            relative_position_index[1:, 1:] = relative_coords.sum(dim=-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer(
                name="relative_position_index",
                tensor=relative_position_index,
            )
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(in_features=all_head_dim, out_features=dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rel_pos_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes multi-head self-attention.

        Args:
            x:
                Input tensor of shape (B, N, C).
            rel_pos_bias:
                Optional shared relative position bias of shape
                (num_heads, N, N) to add to the attention map.

        Returns:
            Output tensor of shape (B, N, C).
        """
        B, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(dim0=-2, dim1=-1)

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1,
            )
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()
            attn = attn + relative_position_bias.unsqueeze(dim=0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(dim0=1, dim1=2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward network (MLP) inside each Transformer block.

    Follows the original BERT implementation: dropout is applied only
    after the second linear layer, not between activation and output.

    Attributes:
        fc1:
            First linear layer projecting from dim to hidden_dim.
        act:
            Activation function (GELU).
        fc2:
            Second linear layer projecting from hidden_dim back to dim.
        drop:
            Dropout layer applied after fc2.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ) -> None:
        """Initializes MLP.

        Args:
            in_features:
                Dimension of the input.
            hidden_features:
                Dimension of the hidden layer. If None, defaults to
                in_features.
            out_features:
                Dimension of the output. If None, defaults to in_features.
            drop:
                Dropout rate applied after the second linear layer.
        """
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(
            in_features=in_features,
            out_features=hidden_features,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(
            in_features=hidden_features,
            out_features=out_features,
        )
        self.drop = nn.Dropout(p=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x:
                Input tensor of shape (B, N, in_features).

        Returns:
            Output tensor of shape (B, N, out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer encoder block.

    Supports optional LayerScale (gamma) for training stability,
    and optional relative position bias.

    Attributes:
        norm1:
            LayerNorm applied before the attention sub-layer.
        attn:
            Multi-head self-attention module.
        drop_path:
            Stochastic depth module.
        norm2:
            LayerNorm applied before the MLP sub-layer.
        mlp:
            Feed-forward network.
        gamma_1:
            LayerScale parameter for the attention branch. None if
            init_values is None or 0.
        gamma_2:
            LayerScale parameter for the MLP branch. None if
            init_values is None or 0.
        layer_idx:
            1-based index of this block in the encoder stack.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        layer_idx: int = 0,
        init_values: Optional[float] = None,
        window_size: Optional[Tuple[int, int]] = None,
        attn_head_dim: Optional[int] = None,
    ) -> None:
        """Initializes TransformerBlock.

        Args:
            dim:
                Dimension of the input and output.
            num_heads:
                Number of attention heads.
            mlp_ratio:
                Ratio of MLP hidden dimension to input dimension.
            qkv_bias:
                If True, enable bias in query and value projections.
            qk_scale:
                Override for attention scaling factor.
            drop:
                Dropout rate for MLP and attention output projections.
            attn_drop:
                Dropout rate for attention weights.
            drop_path_rate:
                Stochastic depth rate for this block.
            layer_idx:
                1-based index of this block (used for weight rescaling).
            init_values:
                If provided and > 0, enables LayerScale with this initial
                value.
            window_size:
                If provided, enables per-head relative position bias.
            attn_head_dim:
                If provided, overrides the computed attention head dimension.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
        )
        self.drop_path = (
            DropPath(drop_prob=drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(normalized_shape=dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )
        self.layer_idx = layer_idx

        if init_values is not None and init_values > 0.0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones(dim),
                requires_grad=True,
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones(dim),
                requires_grad=True,
            )
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(
        self,
        x: torch.Tensor,
        rel_pos_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the block.

        Args:
            x:
                Input tensor of shape (B, N, dim).
            rel_pos_bias:
                Optional shared relative position bias.

        Returns:
            Output tensor of shape (B, N, dim).
        """
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(
                self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class RelativePositionBias(nn.Module):
    """Shared relative position bias across all blocks.

    Computes a learnable relative position bias table that is shared
    by all attention heads in all blocks.

    Attributes:
        window_size:
            Patch grid shape (height, width).
        num_relative_distance:
            Number of unique relative distances plus 3 for CLS token
            interactions.
        relative_position_bias_table:
            Learnable table mapping relative position indices to bias
            values per head.
        relative_position_index:
            Registered buffer mapping 2D relative positions to flat
            indices.
    """

    def __init__(
        self,
        window_size: Tuple[int, int],
        num_heads: int,
    ) -> None:
        """Initializes RelativePositionBias.

        Args:
            window_size:
                Patch grid shape (num_patches_h, num_patches_w).
            num_heads:
                Number of attention heads.
        """
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1
        ) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads)
        )

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, start_dim=1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2,
            dtype=relative_coords.dtype,
        )
        relative_position_index[1:, 1:] = relative_coords.sum(dim=-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.register_buffer(
            name="relative_position_index",
            tensor=relative_position_index,
        )

    def forward(self) -> torch.Tensor:
        """Computes the shared relative position bias.

        Returns:
            Relative position bias of shape
            (num_heads, window_size[0]*window_size[1]+1,
             window_size[0]*window_size[1]+1).
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1,
            -1,
        )
        return relative_position_bias.permute(2, 0, 1).contiguous()


class BEITEncoder(nn.Module):
    """BEIT Vision Transformer encoder.

    The [S] (CLS) token is prepended; 1-D learnable position embeddings
    are added. A learnable mask embedding [M] is used to replace masked
    patches. Supports optional relative position bias and LayerScale.

    Attributes:
        embed_dim:
            Dimension of the token embeddings.
        patch_size:
            Size of each image patch.
        num_patches:
            Total number of patches per image.
        patch_shape:
            Grid shape of patches (height, width).
        patch_embed:
            Patch embedding module.
        cls_token:
            Learnable [S] (CLS) token prepended to the sequence.
        mask_token:
            Learnable [M] token used to replace masked patches.
        pos_embed:
            Learnable 1-D position embeddings. None if use_abs_pos_emb
            is False.
        pos_drop:
            Dropout applied after adding position embeddings.
        rel_pos_bias:
            Shared relative position bias module. None if disabled.
        blocks:
            List of Transformer blocks.
        norm:
            Final LayerNorm.
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
    ) -> None:
        """Initializes BEITEncoder.

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
                Maximum stochastic depth rate (linearly decayed per block).
            init_values:
                If provided and > 0, enables LayerScale with this initial
                value.
            use_abs_pos_emb:
                If True, use learnable absolute position embeddings.
            use_rel_pos_bias:
                If True, use per-block relative position bias.
            use_shared_rel_pos_bias:
                If True, use a single shared relative position bias across
                all blocks.
            attn_head_dim:
                If provided, overrides the computed attention head dimension.
            init_std:
            Standard deviation for truncated normal initialization.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.init_std = init_std

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.patch_shape = self.patch_embed.patch_shape

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, embed_dim)
            )
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_shape,
                num_heads=num_heads,
            )
        else:
            self.rel_pos_bias = None

        dpr = [
            x.item()
            for x in torch.linspace(
                start=0,
                end=drop_path_rate,
                steps=depth,
            )
        ]

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    layer_idx=i + 1,
                    init_values=init_values,
                    window_size=self.patch_shape if use_rel_pos_bias else None,
                    attn_head_dim=attn_head_dim,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes all weights using truncated normal."""
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=self.init_std)
        nn.init.trunc_normal_(self.cls_token, std=self.init_std)
        nn.init.trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights_recursive)
        self.fix_init_weight()

    def _init_weights_recursive(self, m: nn.Module) -> None:
        """Recursive weight initialization applied via nn.Module.apply.

        Args:
            m:
                Module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, val=0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, val=0)
            nn.init.constant_(m.weight, val=1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)

    def fix_init_weight(self) -> None:
        """Rescales output projection weights by layer depth.

        This rescaling is critical for training deep transformers from
        scratch, as described in the BEIT paper.
        """

        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks, start=1):
            rescale(param=layer.attn.proj.weight.data, layer_id=layer_id)
            rescale(param=layer.mlp.fc2.weight.data, layer_id=layer_id)

    def apply_mask(
        self,
        x: torch.Tensor,
        bool_masked_pos: torch.BoolTensor,
    ) -> torch.Tensor:
        """Replaces masked patch positions with the learnable mask token.

        Args:
            x:
                Patch embeddings of shape (B, N, D) without CLS token.
            bool_masked_pos:
                Boolean mask of shape (B, N) where True indicates a masked
                position.

        Returns:
            Masked patch embeddings of shape (B, N, D).
        """
        mask_tokens = self.mask_token.expand(x.shape[0], x.shape[1], -1)
        w = bool_masked_pos.unsqueeze(dim=-1).type_as(mask_tokens)
        x = x * (1 - w) + mask_tokens * w
        return x

    def forward(
        self,
        x: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the encoder.

        Args:
            x:
                Input image tensor of shape (B, C, H, W).
            bool_masked_pos:
                Optional boolean mask of shape (B, N) indicating which
                patch positions are masked. If None, no masking is applied.

        Returns:
            Dictionary with the following keys:
                - 'last_hidden_state': (B, N+1, D) full sequence output
                - 'patch_features': (B, N, D) patch tokens only
                - 'cls_feature': (B, D) CLS token only
        """
        B = x.shape[0]

        tokens = self.patch_embed(x=x)

        if bool_masked_pos is not None:
            tokens = self.apply_mask(x=tokens, bool_masked_pos=bool_masked_pos)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for block in self.blocks:
            tokens = block(x=tokens, rel_pos_bias=rel_pos_bias)

        tokens = self.norm(tokens)

        return {
            "last_hidden_state": tokens,
            "patch_features": tokens[:, 1:],
            "cls_feature": tokens[:, 0],
        }
