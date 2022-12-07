from __future__ import annotations
from functools import partial
import math
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from lightly.models import utils

# vision_transformer requires torchvision >= 0.12
from torchvision.models import vision_transformer
from torchvision.models.vision_transformer import ConvStemConfig


class MAEEncoder(vision_transformer.Encoder):
    """Encoder for the Masked Autoencoder model [0].

    Encodes patch embeddings. Code inspired by [1].

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377
    - [1]: https://github.com/facebookresearch/mae

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
    def from_vit_encoder(cls, vit_encoder: vision_transformer.Encoder) -> MAEEncoder:
        """Creates a MAEEncoder from a torchvision ViT encoder."""
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
        encoder.pos_embedding = vit_encoder.pos_embedding
        encoder.dropout = vit_encoder.dropout
        encoder.layers = vit_encoder.layers
        encoder.ln = vit_encoder.ln
        return encoder

    def forward(
        self, 
        input: torch.Tensor, 
        idx_keep: Optional[torch.Tensor] = None
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
            pos_embedding.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embedding = pos_embedding.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embedding), dim=1)


class MAEBackbone(vision_transformer.VisionTransformer):
    """Backbone for the Masked Autoencoder model [0].

    Converts images into patches and encodes them. Code inspired by [1]. 
    Note that this implementation uses a learned positional embedding while [0]
    uses a fixed positional embedding.

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377
    - [1]: https://github.com/facebookresearch/mae
    - [2]: Early Convolutions Help Transformers See Better, 2021, https://arxiv.org/abs/2106.14881.

    Attributes:
        image_size:
            Input image size.
        patch_size:
            Width and height of the image patches. image_size must be a multiple
            of patch_size.
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
        num_classes:
            Number of classes for the classification head. Currently not used.
        representation_size:
            If specified, an additional linear layer is added before the 
            classification head to change the token dimension from hidden_dim
            to representation_size. Currently not used.
        norm_layer:
            Callable that creates a normalization layer.
        conv_stem_configs:
            If specified, a convolutional stem is added at the beggining of the
            network following [2]. Not used in the original Masked Autoencoder
            paper [0].

    """
    def __init__(
        self, 
        image_size: int, 
        patch_size: int, 
        num_layers: int, 
        num_heads: int, 
        hidden_dim: int, 
        mlp_dim: int, 
        dropout: float = 0, 
        attention_dropout: float = 0, 
        num_classes: int = 1000, 
        representation_size: Optional[int] = None, 
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
            representation_size=representation_size,
            norm_layer=norm_layer,
            conv_stem_configs=conv_stem_configs,
        )
        self.encoder = MAEEncoder(
            seq_length=self.seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )

    @classmethod
    def from_vit(cls, vit: vision_transformer.VisionTransformer) -> MAEBackbone:
        """Creates a MAEBackbone from a torchvision ViT model."""
        # Create a new instance with dummy values as they will be overwritten
        # by the copied vit_encoder attributes
        backbone = cls(
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
        backbone.conv_proj = vit.conv_proj
        backbone.class_token = vit.class_token
        backbone.seq_length = vit.seq_length
        backbone.heads = vit.heads
        backbone.encoder = MAEEncoder.from_vit_encoder(vit.encoder)
        return backbone

    def forward(
        self, 
        images: torch.Tensor, 
        idx_keep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returns encoded class tokens from a batch of images.

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
        out = self.encode(images, idx_keep)
        class_token = out[:, 0]
        return class_token

    def encode(
        self, 
        images: torch.Tensor, 
        idx_keep: Optional[torch.Tensor] = None
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
        out = self.images_to_tokens(images, prepend_class_token=True)
        return self.encoder(out, idx_keep)


    def images_to_tokens(self, images: torch.Tensor, prepend_class_token: bool) -> torch.Tensor:
        """Converts images into patch tokens.
        
        Args:
            images:
                Tensor with shape (batch_size, channels, image_size, image_size).
        
        Returns:
            Tensor with shape (batch_size, sequence_length - 1, hidden_dim)
            containing the patch tokens.
        """
        x = self.conv_proj(images)
        tokens =  x.flatten(2).transpose(1, 2)
        if prepend_class_token:
            tokens = utils.prepend_class_token(tokens, self.class_token)
        return tokens


class MAEDecoder(vision_transformer.Encoder):
    """Decoder for the Masked Autoencoder model [0].

    Decodes encoded patches and predicts pixel values for every patch.
    Code inspired by [1].

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377
    - [1]: https://github.com/facebookresearch/mae

    Attributes:
        seq_length:
            Token sequence length, including the class token.
        num_layers:
            Number of transformer blocks.
        num_heads:
            Number of attention heads.
        embed_input_dim:
            Dimension of the input tokens. Usually be equal to the hidden
            dimension of the MAEEncoder or MAEBackbone.
        hidden_dim:
            Dimension of the decoder tokens.
        mlp_dim:
            Dimension of the MLP in the transformer block.
        out_dim:
            Output dimension of the prediction for a single patch. Usually equal
            to (3 * patch_size ** 2).
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
        embed_input_dim: int,
        hidden_dim: int,
        mlp_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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

    def embed(self, input: torch.Tensor) -> torch.Tensor:
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
        return self.decoder_embed(input)

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder transformer.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, hidden_dim) containing
                the encoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, hidden_dim) containing
            the decoded tokens.

        """
        return super().forward(input)

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        """Predics pixel values from decoded tokens.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, hidden_dim) containing
                the decoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, out_dim) containing
            predictions for each token.

        """
        return self.prediction_head(input)
