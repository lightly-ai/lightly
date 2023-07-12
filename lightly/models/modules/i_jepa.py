import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.models import vision_transformer

from lightly.models import utils
from typing import Optional, Callable
from functools import partial
import math


class IJEPA_predictor(vision_transformer.Encoder):
    """
    Predictor for the I-JEPA model [0].

    Predict patch embeddings. Code inspired by [1].

    - [0]: Joint-Embedding Predictive Architecture, 2023, https://arxiv.org/abs/2301.08243
    - [1]: https://github.com/facebookresearch/ijepa

    Attributes:
        seq_length:
            Token sequence length, including the class token.
        num_layers:
            Number of transformer blocks.
        num_heads:
            Number of attention heads.
        hidden_dim:
            Dimension of the input and output tokens.
        predictor_embed_dim:
            Dimension of inner predicted tokens
        mlp_dim:
            Dimension of the MLP in the transformer block.
        dropout:
            Percentage of elements set to zero after the MLP in the transformer.
        attention_dropout:
            Percentage of elements set to zero after the attention head.

    """
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        predictor_embed_dim :int, 
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    ):
        super().__init__(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )
        self.predictor_embed = nn.Linear(mlp_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.predictor_proj = nn.Linear(predictor_embed_dim, mlp_dim, bias=True)
        # self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim),
        #                                         requires_grad=False)
        # predictor_pos_embed = get_2d_sincos_pos_embed(self.predictor_pos_embed.shape[-1],
        #                                               int(num_patches**.5),
        #                                               cls_token=False)
        # self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))

    @classmethod
    def from_vit_encoder(cls, vit_encoder):
        """Creates a I-JEPA predictor backbone (mhas and layernorm) from a torchvision ViT encoder."""
        # Create a new instance with dummy values as they will be overwritten
        # by the copied vit_encoder attributes
        encoder = cls(
            seq_length=1,
            num_layers=1,
            num_heads=1,
            hidden_dim=1,
            mlp_dim=1,
            dropout=0,
            attention_dropout=0,
        )
        encoder.layers = vit_encoder.layers
        encoder.ln = vit_encoder.ln
        return encoder

    def forward(
        self,  x, masks_x, masks
    ):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        B = len(x) // len(masks_x)

        x = self.predictor_embed(x)

        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += utils.apply_masks(x_pos_embed, masks_x)
        _, N_ctxt, _ = x.shape

        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = utils.apply_masks(pos_embs, masks)
        pos_embs = utils.repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)

        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        x = self.ln(self.layers(x))

        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x

        
class IJEPA_encoder(vision_transformer.Encoder):
    """
    Encoder for the I-JEPA model [0].

    Encodes patch embeddings. Code inspired by [1].

    - [0]: Joint-Embedding Predictive Architecture, 2023, https://arxiv.org/abs/2301.08243
    - [1]: https://github.com/facebookresearch/ijepa

    Attributes:
        seq_length:
            Token sequence length, including the class token.
        num_layers:
            Number of transformer blocks.
        num_heads:
            Number of attention heads.
        hidden_dim:
            Dimension of the input and output tokens.
        mlp_dim:
            Dimension of the MLP in the transformer block.
        dropout:
            Percentage of elements set to zero after the MLP in the transformer.
        attention_dropout:
            Percentage of elements set to zero after the attention head.

    """

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )

    @classmethod
    def from_vit_encoder(cls, vit_encoder: vision_transformer.Encoder):
        """Creates a MAEEncoder from a torchvision ViT encoder."""
        # Create a new instance with dummy values as they will be overwritten
        # by the copied vit_encoder attributes
        encoder = cls(
            seq_length=1,
            num_layers=1,
            num_heads=1,
            hidden_dim=1,
            mlp_dim=1,
            dropout=0,
            attention_dropout=0,
        )
        encoder.dropout = vit_encoder.dropout
        encoder.layers = vit_encoder.layers
        encoder.ln = vit_encoder.ln
        return encoder

    def forward(
        self, input: torch.Tensor, idx_keep: Optional[torch.Tensor] = None
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
            Batch of encoded output tokens.
        """
        input = input + self.interpolate_pos_encoding(input)
        if idx_keep is not None:
            input = utils.get_at_index(input, idx_keep)
        return self.ln(self.layers(self.dropout(input)))

    def interpolate_pos_encoding(self, input: torch.Tensor):
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
        N = self.pos_embedding.shape[1] - 1
        if npatch == N:
            return self.pos_embedding
        class_emb = self.pos_embedding[:, 0]
        pos_embedding = self.pos_embedding[:, 1:]
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