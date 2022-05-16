from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torchvision

from lightly.models import utils

# TODO: Check for correct torchvision version

class MAEEncoder(torchvision.models.vision_transformer.Encoder):

    def encode(self, input: torch.Tensor, idx_keep: Optional[torch.Tensor] = None):
        input = input + self.pos_embedding
        if idx_keep is not None:
            input = utils.get_at_index(input, idx_keep)
        return self.ln(self.layers(self.dropout(input)))

    @classmethod
    def from_vit_encoder(cls, vit_encoder):
        # Create encoder instance with dummy values as they will be overwritten
        # by the copied vit_encoder attributes.
        encoder = cls(
            seq_length=1,
            num_layers=1,
            num_heads=1,
            hidden_dim=1,
            mlp_dim=1,
            dropout=0,
            attention_dropout=0,
        )
        encoder.pos_embedding = vit_encoder.pos_embedding
        encoder.dropout = vit_encoder.dropout
        encoder.layers = vit_encoder.layers
        encoder.ln = vit_encoder.ln
        return encoder


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
        self.decoder_embed = nn.Linear(embed_input_dim, hidden_dim, bias=True)
        self.prediction_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, input):
        return self.decode(input)

    def embed(self, input):
        return self.decoder_embed(input)

    def decode(self, input):
        return super().forward(input)

    def predict(self, input):
        return self.prediction_head(input)
