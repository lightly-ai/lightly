from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from torch import Tensor
from torch.nn import LayerNorm, Module, Parameter, Sequential

from lightly.models import utils
from lightly.models.modules.masked_vision_transformer_timm import init_weights


class MAEDecoderTIMM(Module):
    """Decoder for the Masked Autoencoder model [0].

    Decodes encoded patches and predicts pixel values for every patch.
    Code inspired by [1].

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377
    - [1]: https://github.com/facebookresearch/mae

    Attributes:
        num_patches:
            Number of patches.
        patch_size:
            Patch size.
        in_chans:
            Number of image input channels.
        embed_dim:
            Embedding dimension of the encoder.
        decoder_embed_dim:
            Embedding dimension of the decoder.
        decoder_depth:
            Depth of transformer.
        decoder_num_heads:
            Number of attention heads.
        mlp_ratio:
            Ratio of mlp hidden dim to embedding dim.
        proj_drop_rate:
            Percentage of elements set to zero after the MLP in the transformer.
        attn_drop_rate:
            Percentage of elements set to zero after the attention head.
        norm_layer:
            Normalization layer.
        initialize_weights:
            Flag that determines if weights should be initialized.
        mask_token:
            The mask token.

    """

    def __init__(
        self,
        num_patches: int,
        patch_size: int,
        in_chans: int = 3,
        embed_dim: int = 1024,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(LayerNorm, eps=1e-6),
        initialize_weights: bool = True,
        mask_token: Optional[Parameter] = None,
    ):
        """Initializes the MAEDecoderTIMM with the specified parameters."""

        super().__init__()

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            if mask_token is None
            else mask_token
        )

        # Positional encoding of the decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = Sequential(
            *[
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch

        if initialize_weights:
            self._initialize_weights()

    def forward(self, input: Tensor) -> Tensor:
        """Returns predicted pixel values from encoded tokens.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, embed_input_dim).

        Returns:
            Tensor with shape (batch_size, seq_length, out_dim).
        """

        out = self.embed(input)
        out = self.decode(out)
        return self.predict(out)

    def embed(self, input: Tensor) -> Tensor:
        """Embeds encoded input tokens into decoder token dimension.

        This is a single linear layer that changes the token dimension from
        embed_input_dim to hidden_dim.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, embed_input_dim)
                containing the encoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, hidden_dim) containing
            the embedded tokens.

        """
        out: Tensor = self.decoder_embed(input)
        return out

    def decode(self, input: Tensor) -> Tensor:
        """Forward pass through the decoder transformer.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, hidden_dim) containing
                the encoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, hidden_dim) containing
            the decoded tokens.

        """
        output: Tensor = input + self.decoder_pos_embed
        output = self.decoder_blocks(output)
        output = self.decoder_norm(output)
        return output

    def predict(self, input: Tensor) -> Tensor:
        """Predicts pixel values from decoded tokens.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, hidden_dim) containing
                the decoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, out_dim) containing
            predictions for each token.

        """
        out: Tensor = self.decoder_pred(input)
        return out

    def _initialize_weights(self) -> None:
        """Initializes weights for the decoder components."""

        torch.nn.init.normal_(self.mask_token, std=0.02)
        utils.initialize_2d_sine_cosine_positional_embedding(
            pos_embedding=self.decoder_pos_embed, has_class_token=True
        )
        self.apply(init_weights)
