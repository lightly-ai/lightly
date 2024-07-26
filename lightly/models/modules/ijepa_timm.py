import math
from functools import partial
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from lightly.models import utils


def apply_masks(
    x: torch.Tensor, masks: Union[torch.Tensor, List[torch.Tensor]]
) -> torch.Tensor:
    """
    Apply masks to the input tensor.

    Args:
        x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
        masks: tensor or list of tensors containing indices of patches in [N] to keep
    Returns:
        tensor of shape [B, N', D] where N' is the number of patches to keep
    """
    if not isinstance(masks, list):
        masks = [masks]

    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


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
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
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
        assert (masks is not None) and (
            masks_x is not None
        ), "Cannot run predictor without mask indices"

        B = len(x) // len(masks_x)
        x = self.predictor_embed(x)
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)

        x += apply_masks(x_pos_embed, masks_x)
        _, N_ctxt, _ = x.shape

        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = self.repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)

        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
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


class IJEPAEncoderTIMM(nn.Module):
    """Encoder for the I-JEPA model [0].

    Experimental: Support for I-JEPA is experimental, there might be breaking changes
    in the future.

    Encodes patch embeddings. Code inspired by [1].

    - [0]: Joint-Embedding Predictive Architecture, 2023, https://arxiv.org/abs/2301.08243
    - [1]: https://github.com/facebookresearch/ijepa

    Attributes:
        num_patches:
            Number of patches (tokens), including the class token.
        depth:
            Number of transformer blocks.
        mlp_dim:
            Dimension of the MLP in the transformer block.
        embed_dim:
            Dimension of the input and output tokens.
        num_heads:
            Number of attention heads.
        qkv_bias:
            If True, add bias to the query, key, and value tensors.
        mlp_ratio:
            Ratio of mlp hidden dim to embedding dim.
        drop_rate:
            Percentage of elements set to zero after the MLP in the transformer.
        attn_drop_rate:
            Percentage of elements set to zero after the attention head.
        norm_layer:
            Callable that creates a normalization layer.

    """

    def __init__(
        self,
        num_patches: int,
        depth: int,
        num_heads: int,
        embed_dim: int,
        drop_rate: float,
        attn_drop_rate: float,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )
        pos_embed = utils.get_2d_sincos_pos_embed(
            embed_dim, int(num_patches**0.5), cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(
        self,
        input: torch.Tensor,
        idx_keep: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Encode input tokens.

        Args:
            input:
                Batch of token sequences.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                If specified, only the indexed tokens will be encoded.

        Returns:
            Batch of encoded output tokens.s
        """
        input = input + self.interpolate_pos_encoding(input)
        if idx_keep is not None:
            input = apply_masks(input, idx_keep)

        for blk in self.blocks:
            input = blk(input)

        input = self.norm(input)
        return input

    def interpolate_pos_encoding(self, input: torch.Tensor) -> torch.Tensor:
        """Returns the interpolated positional embedding for the given input.

        This function interpolates self.pos_embedding for all tokens in the input,
        ignoring the class token. This allows encoding variable sized images.

        Args:
            input:
               Input tensor with shape (batch_size, num_sequences).

        """
        # code copied from:
        # https://github.com/facebookresearch/msn/blob/4388dc1eadbe3042b85d3296d41b9b207656e043/src/deit.py#L291
        npatch = input.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N:
            return self.pos_embed
        class_emb = self.pos_embed[:, 0]
        pos_embedding = self.pos_embed[:, 1:]
        dim = input.shape[-1]
        pos_embedding = nn.functional.interpolate(
            pos_embedding.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=math.sqrt(npatch / N),
            mode="bicubic",
        )
        pos_embedding = pos_embedding.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embedding), dim=1)


class IJEPABackboneTIMM(nn.Module):
    """Encoder for the I-JEPA model [0].

    Experimental: Support for I-JEPA is experimental, there might be breaking changes
    in the future.

    Converts images into patches and encodes them. Code inspired by [1].
    Note that this implementation uses a learned positional embedding while [0]
    uses a fixed positional embedding.

    - [0]: Joint-Embedding Predictive Architecture, 2023, https://arxiv.org/abs/2301.08243
    - [1]: https://github.com/facebookresearch/ijepa

    Attributes:
        image_size:
            Input image size.
        in_channels:
            Number of input image channels.
        patch_size:
            Width and height of the image patches. image_size must be a multiple
            of patch_size.
        embed_dim:
            Dimension of the input and output tokens.
        depth:
            Number of transformer blocks.
        num_heads:
            Number of attention heads.
        mlp_ratio:
            Ratio of mlp hidden dim to embedding dim.
        drop_rate:
            Percentage of elements set to zero after the MLP in the transformer.
        attn_drop_rate:
            Percentage of elements set to zero after the attention head.
        norm_layer:
            Callable that creates a normalization layer.

    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: int = 4,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.encoder = IJEPAEncoderTIMM(
            num_patches,
            depth,
            num_heads,
            embed_dim,
            drop_rate,
            attn_drop_rate,
            mlp_ratio,
            norm_layer=norm_layer,
        )

        self.conv_proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(
        self, images: torch.Tensor, idx_keep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns encoded class tokens from a batch of images.

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
        out = self.encode(images, idx_keep)  # type: torch.Tensor
        return out

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
        out = self.images_to_tokens(images)  # type: torch.Tensor
        out = self.encoder(out, idx_keep)
        return out

    def images_to_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Converts images into patch tokens.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_size, image_size).

        Returns:
            Tensor with shape (batch_size, sequence_length - 1, hidden_dim)
            containing the patch tokens.
        """
        x: torch.Tensor = self.conv_proj(images)
        tokens = x.flatten(2).transpose(1, 2)
        return tokens
