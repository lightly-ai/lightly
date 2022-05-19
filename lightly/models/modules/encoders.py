from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torchvision
from torchvision.models import vision_transformer

from lightly.models import utils

# TODO: Check for correct torchvision version


import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_pos_embedding(seq_length, hidden_dim):
    grid_size = int((seq_length - 1)**.5)
    pos_embed = get_2d_sincos_pos_embed(hidden_dim, grid_size, cls_token=True)
    return nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=False)

class MAEEncoder(vision_transformer.Encoder):

    @classmethod
    def from_vit_encoder(cls, vit_encoder: vision_transformer.Encoder):
        # Create a new instance with dummy values as they will be overwritten
        # by the copied vit_encoder attributes
        encoder = cls(
            seq_length=1,
            num_layers=1,
            num_heads=1,
            hidden_dim=1,
            mlp_dim=1,
            dropout=0,
            attention_dropout=0,
        )
        # encoder.pos_embedding = vit_encoder.pos_embedding
        _, seq_length, hidden_dim = vit_encoder.pos_embedding.shape
        encoder.pos_embedding = get_pos_embedding(seq_length, hidden_dim)
        encoder.dropout = vit_encoder.dropout
        encoder.layers = vit_encoder.layers
        encoder.ln = vit_encoder.ln
        return encoder

    def forward(self, input: torch.Tensor, idx_keep: Optional[torch.Tensor] = None):
        """Encode input tokens. If idx_keep is specified only the tokens with
        the provided indices are encoded.
        """
        input = input + self.pos_embedding
        if idx_keep is not None:
            input = utils.get_at_index(input, idx_keep)
        return self.ln(self.layers(self.dropout(input)))


class MAEBackbone(torchvision.models.vision_transformer.VisionTransformer):
    @classmethod
    def from_vit(cls, vit: torchvision.models.vision_transformer.VisionTransformer):
        # Create a new instance with dummy values as they will be overwritten
        # by the copied vit_encoder attributes
        encoder = cls(
            image_size=vit.image_size,
            patch_size=vit.patch_size,
            num_layers=1,
            num_heads=1,
            hidden_dim=vit.hidden_dim,
            mlp_dim=vit.mlp_dim,
            dropout=vit.dropout,
            attention_dropout=vit.attention_dropout,
            num_classes=vit.num_classes,
            representation_size=vit.representation_size,
            norm_layer=vit.norm_layer,
        )
        encoder.conv_proj = vit.conv_proj
        encoder.class_token = vit.class_token
        encoder.seq_length = vit.seq_length
        encoder.heads = vit.heads
        encoder.encoder = MAEEncoder.from_vit_encoder(vit.encoder)
        return encoder

    def forward(self, images, idx_keep: Optional[torch.Tensor] = None):
        """Returns encoded class tokens from images."""
        out = self.encode(images, idx_keep)
        class_token = out[:, 0]
        return class_token

    def encode(self, images, idx_keep: Optional[torch.Tensor] = None):
        """Returns encoded class and patch tokens from images."""
        out = self.images_to_patch_embeddings(images)
        out = utils.prepend_class_token(out, self.class_token)
        return self.encoder(out, idx_keep)

    def images_to_patch_embeddings(self, images: torch.Tensor):
        """Converts images into patch embeddings."""
        # output has shape (batch_size, height_n_patches * width_n_patches, embed_dim)
        x = self.conv_proj(images)
        return x.flatten(2).transpose(1, 2) 


class MAEDecoder(torchvision.models.vision_transformer.Encoder):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        embed_input_dim: int,
        hidden_dim: int,
        mlp_dim: int,
        out_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
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
        self.pos_embedding = get_pos_embedding(seq_length, hidden_dim)
        self.decoder_embed = nn.Linear(embed_input_dim, hidden_dim, bias=True)
        self.prediction_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, input):
        """Returns predicted pixel values from encoded input tokens."""
        out = self.embed(input)
        out = self.decode(input)
        return self.predict(out)

    def embed(self, input):
        """Converts encoded input tokens into decoder tokens with 
        size self.hidden_dimension.
        """
        return self.decoder_embed(input)

    def decode(self, input):
        """Decodes encoded and embedded input tokens."""
        return super().forward(input)

    def predict(self, input):
        """Predics pixel values from decoded input tokens."""
        return self.prediction_head(input)
