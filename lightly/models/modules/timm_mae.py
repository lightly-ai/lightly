from __future__ import annotations

import math
from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.nn import Linear, Module, Parameter

from torchvision.models.vision_transformer import ConvStemConfig

from timm.models import vision_transformer

from lightly.models import utils



class MAEBackbone(vision_transformer.VisionTransformer):
    """Backbone for the Masked Autoencoder model [0].

    Converts images into patches and encodes them. Code inspired by [1].
    Note that this implementation uses a fixed positional embedding.

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
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__(
            img_size=image_size, 
            patch_size=patch_size,
            depth=num_layers,
            num_heads=num_heads,
            embed_dim=hidden_dim,
            mlp_ratio=mlp_dim/hidden_dim,
            drop_rate=dropout,
            attn_drop_rate=attention_dropout, 
            num_classes=num_classes,
            norm_layer=norm_layer,
        )


    @classmethod
    def from_vit(
        cls, vit: vision_transformer.VisionTransformer, initialize_weights: bool = True
    ) -> MAEBackbone:
        """Creates a MAEBackbone from a timm ViT model.

        Args:
            vit:
                A timm ViT model.
            initialize_weights:
                If True, then all weights are initialized as in MAE paper. Set this to
                False if vit is pretrained.

        Returns:
            A MAEBackbone with the same architecture as vit.

        """
        # Create a new instance with dummy values as they will be overwritten
        # by the copied vit_encoder attributes
        
        backbone = cls(
            img_size=1,
            patch_size=1, 
            num_layers=1,
            num_heads=1,
            embed_dim=vit.embed_dim,
            mlp_dim=1,
            dropout=0.2,
            attention_dropout=0.2,
            num_classes=vit.num_classes,
            representation_size=16,
            norm_layer=vit.norm,
        )
        backbone.patch_embed = vit.patch_embed
        backbone.blocks = vit.blocks
        return backbone

    def forward(
        self, images: torch.Tensor, idx_keep: Optional[torch.Tensor] = None
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
        # convert images to tokens and add class token if needed
        input = self.images_to_tokens(images, prepend_class_token=True)
        # add positional encoding
        input = input + self.interpolate_pos_encoding(input)
        # get the tokens that are kept  
        if idx_keep is not None:
            input = utils.get_at_index(input, idx_keep)    
        # apply Transformer blocks
        for blk in self.blocks:
            input = blk(input)
        # normalize
        out = self.norm(input)
        return out
    
    def interpolate_pos_encoding(self, input: torch.Tensor):
        """Returns the interpolated positional embedding for the given input.

        This function interpolates self.pos_embed for all tokens in the input,
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
        pos_embed = self.pos_embed[:, 1:]
        dim = input.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=math.sqrt(npatch / N),
            mode="bicubic",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)

    def images_to_tokens(
        self, images: torch.Tensor, prepend_class_token: bool
    ) -> torch.Tensor:
        """Converts images into patch tokens.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_size, image_size).

        Returns:
            Tensor with shape (batch_size, sequence_length - 1, hidden_dim)
            containing the patch tokens.
        """
        tokens = self.patch_embed(images)
        if prepend_class_token:
            tokens = utils.prepend_class_token(tokens, self.cls_token)
        return tokens

    def _initialize_weights(self) -> None:
        # Initialize the patch embedding layer like a linear layer instead of conv
        # layer.
        w = self.conv_proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize the class token.
        torch.nn.init.normal_(self.class_token, std=0.02)

        self.encoder._initialize_weights()
        _initialize_2d_sine_cosine_positional_embedding(self.pos_embed)
        _initialize_linear_layers(self)


class MAEDecoder(nn.Module):
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
        num_patches = seq_length - 1
        
        self.decoder_embed = nn.Linear(embed_input_dim, hidden_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # positional encoding of the decoder
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim), requires_grad=False)  # fixed sin-cos embedding

        mlp_ratio = mlp_dim / hidden_dim
        self.decoder_blocks = nn.ModuleList([
            vision_transformer.Block(hidden_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(num_layers)])

        self.decoder_norm = norm_layer(hidden_dim)
        self.decoder_pred = nn.Linear(hidden_dim, out_dim, bias=True) # decoder to patch
        
        self._initialize_weights()

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
        input = input + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            input = blk(input)
        input = self.decoder_norm(input)
        return input

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

    def _initialize_weights(self) -> None:
        _initialize_2d_sine_cosine_positional_embedding(self.pos_embedding)
        _initialize_linear_layers(self)


def _initialize_2d_sine_cosine_positional_embedding(pos_embedding: Parameter) -> None:
    _, seq_length, hidden_dim = pos_embedding.shape
    grid_size = int((seq_length - 1) ** 0.5)
    sine_cosine_embedding = utils.get_2d_sine_cosine_positional_embedding(
        embed_dim=hidden_dim,
        grid_size=grid_size,
        cls_token=True,
    )
    pos_embedding.data.copy_(
        torch.from_numpy(sine_cosine_embedding).float().unsqueeze(0)
    )
    # Freeze positional embedding.
    pos_embedding.requires_grad = False


def _initialize_linear_layers(module: Module) -> None:
    def init(mod: Module) -> None:
        if isinstance(mod, Linear):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0)

    module.apply(init)
