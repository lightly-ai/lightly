from __future__ import annotations

import math
from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.nn import Linear, Module, Parameter

# # vision_transformer requires torchvision >= 0.12
# from torchvision.models import vision_transformer
from torchvision.models.vision_transformer import ConvStemConfig
from timm.models.vision_transformer import PatchEmbed, Block

from lightly.models import utils


class MAEEncoder(nn.Module):
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
        #TODO:check if partial is necessary here
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
        self.mlp_ratio = mlp_dim / hidden_dim 
        # proj_drop and attn_drop of Block are set to zero in the mae implementation, so
        # we ignore the values of dropout and attention_dropout
        self.blocks = nn.ModuleList([
            Block(dim=hidden_dim, num_heads=num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(num_layers)])
        self.norm = norm_layer(hidden_dim)
        self._initialize_weights()

    @classmethod
    def from_vit_encoder(
        cls, vit_encoder_blocks: nn.ModuleList, vit_encoder_norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), initialize_weights: bool = True
    ) -> MAEEncoder:
        """Creates a MAEEncoder from a torchvision ViT encoder.

        Args:
            vit_encoder_blocks:
                A module list of TIMM Blocks (https://github.com/huggingface/pytorch-image-models/blob/e0079c92da51319be2eb380fbd3539160acee320/timm/models/vision_transformer.py#L123).
            vit_encoder_norm_layer:
                A nn.LayerNorm layer.
            initialize_weights:
                If True, then all weights are initialized as in MAE paper. Set this to
                False if vit_encoder is pretrained.

        Returns:
            A MAEEncoder with the architecture specified by the TIMM blocks and norm layer.

        """
        # Create a new instance with dummy values as they will be overwritten
        # by the copied vit_encoder attributes
        encoder = cls(
            seq_length=197,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            dropout=0,
            attention_dropout=0,
        )
        encoder.blocks = vit_encoder_blocks
        encoder.norm = vit_encoder_norm_layer
        if initialize_weights:
            encoder._initialize_weights()
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
        # commenting out for now as this is not present at the original MAE implementation
        # input = input + self.interpolate_pos_encoding(input)
        if idx_keep is not None:
            input = utils.get_at_index(input, idx_keep)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

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

    def _initialize_weights(self) -> None:
        _initialize_2d_sine_cosine_positional_embedding(self.pos_embedding)
        _initialize_linear_layers(self)


class MAEBackbone(nn.Module):
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
        conv_stem_configs: Optional[List[ConvStemConfig]] = None, #should be None for MAE - PatchEmbed has 1 conv layer
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
        # don't set in_chans <=> keeps default value in_chans=3 as in prev_channels of 
        # VisionTransformer see:
        # https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, embed_dim=hidden_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim), requires_grad=False)  # fixed sin-cos embedding
        self.seq_length = num_patches + 1
        
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
    def from_vit(
        cls, patch_embed: PatchEmbed, vit_encoder_blocks: nn.ModuleList, vit_encoder_norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), initialize_weights: bool = True
    ) -> MAEBackbone:
        """Creates a MAEBackbone from a ViT model defined as a module list.

        Args:
            vit:
                A ViT model.
            initialize_weights:
                If True, then all weights are initialized as in MAE paper. Set this to
                False if vit is pretrained.

        Returns:
            A MAEBackbone with the same architecture as vit.

        """
        # Create a new instance with dummy values as they will be overwritten
        # by the copied vit_encoder attributes
        backbone = cls(
            image_size=256,
            patch_size=16,
            num_layers=1,
            num_heads=1,
            hidden_dim=768,
            mlp_dim=3072,
            dropout=0.,
            attention_dropout=0.,
            num_classes=1000,
            representation_size=1024,
            norm_layer=torch.nn.LayerNorm,
        )
        # must figure out how to get the embedding dimension
        # first_block = vit_encoder_blocks.pop(0)
        # embed_dim = first_block.dim
        backbone.patch_embed = PatchEmbed(img_size=patch_embed.img_size, 
                                          patch_size=patch_embed.patch_size, 
                                          embed_dim=patch_embed.embed_dim)
        backbone.encoder = MAEEncoder.from_vit_encoder(
            vit_encoder_blocks, vit_encoder_norm_layer, initialize_weights=initialize_weights
        )
        backbone.cls_token = nn.Parameter(torch.zeros(1, 1, patch_embed.embed_dim))
        
        backbone.pos_embed = nn.Parameter(torch.zeros(1, patch_embed.num_patches + 1, patch_embed.dim), requires_grad=False)  # fixed sin-cos embedding
        backbone.seq_length = patch_embed.num_patches + 1
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
        out = self.images_to_tokens(images, prepend_class_token=True)
        return self.encoder(out, idx_keep)

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
        # embed patches
        x = self.patch_embed(images)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        if prepend_class_token:
            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # x = self.conv_proj(images)
        # tokens = x.flatten(2).transpose(1, 2)
        # if prepend_class_token:
        #     tokens = utils.prepend_class_token(tokens, self.class_token)
        return x

    def _initialize_weights(self) -> None:
        # Initialize the patch embedding layer like a linear layer instead of conv
        # layer.
        w = self.conv_proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize the class token.
        torch.nn.init.normal_(self.class_token, std=0.02)

        self.encoder._initialize_weights()
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
        # self.decoder_embed = nn.Linear(embed_input_dim, hidden_dim, bias=True)
        # self.prediction_head = nn.Linear(hidden_dim, out_dim)
        # 
        num_patches = seq_length - 1
        mlp_ratio = mlp_dim / hidden_dim 
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_input_dim, hidden_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(hidden_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(num_layers)])

        self.decoder_norm = norm_layer(hidden_dim)
        # ensure that: out_dim = patch_size**2 * in_chans
        self.prediction_head = nn.Linear(hidden_dim, out_dim, bias=True) # decoder to patch
        self.norm_pix_loss = False # hardcode to False for now
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
        out = self.predict(out)
        # remove cls token
        out = out[:, 1:, :]
        return out

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
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(input.shape[0], ids_restore.shape[1] + 1 - input.shape[1], 1)
        input_ = torch.cat([input[:, 1:, :], mask_tokens], dim=1)  # no cls token
        input_ = torch.gather(input_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, input.shape[2]))  # unshuffle
        input = torch.cat([input[:, :1, :], input_], dim=1)  # append cls token

        # add pos embed
        x = input + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return x

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
