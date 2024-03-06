import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, Module, Parameter
from torchvision.models.vision_transformer import VisionTransformer

from lightly.models import utils
from lightly.models.modules.masked_vision_transformer import MaskedVisionTransformer


class MaskedVisionTransformerTorchvision(MaskedVisionTransformer, Module):
    """Masked Vision Transformer class using Torchvision.

    Attributes:
        vit:
            The VisionTransformer object of Torchvision.
        mask_token:
            The mask token.

    """

    def __init__(
        self,
        vit: VisionTransformer,
        mask_token: Optional[Parameter] = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.mask_token = (
            mask_token
            if mask_token is not None
            else Parameter(torch.zeros(1, 1, self.vit.hidden_dim))
        )
        self._initialize_weights()

    @property
    def sequence_length(self) -> int:
        seq_len: int = self.vit.seq_length
        return seq_len

    def interpolate_pos_encoding(self, input: Tensor) -> Tensor:
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
        N = self.vit.encoder.pos_embedding.shape[1] - 1
        if npatch == N:
            pos_embedding: Tensor = self.vit.encoder.pos_embedding
            return pos_embedding
        class_emb = self.vit.encoder.pos_embedding[:, 0]
        pos_embedding = self.vit.encoder.pos_embedding[:, 1:]
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

    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns encoded class tokens from a batch of images.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_size, image_size).
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                If specified, the indexed tokens are masked with self.mask_token.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                If specified, only the indexed tokens will be passed to the
                encoder.

        Returns:
            Tensor with shape (batch_size, vit.hidden_dim) containing the
            encoded class token for every image.

        """
        out = self.encode(images, idx_mask=idx_mask, idx_keep=idx_keep)
        class_token = out[:, 0]
        return class_token

    def encode(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode input images.

        Args:
            input:
                Batch of input images.
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                If specified, the indexed tokens are masked with self.mask_token.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                If specified, only the indexed tokens will be encoded.

        Returns:
            Batch of encoded output tokens.
        """
        # convert images to tokens
        input = self.images_to_tokens(images)
        # add prefix tokens if needed
        input = self.add_prefix_tokens(input)

        if idx_mask is not None:
            input = utils.mask_at_index(input, idx_mask, self.mask_token)
        # add positional encoding
        input = self.add_pos_embed(input)

        if idx_keep is not None:
            input = utils.get_at_index(input, idx_keep)
        out: Tensor = self.vit.encoder.ln(
            self.vit.encoder.layers(self.vit.encoder.dropout(input))
        )
        return out

    def images_to_tokens(self, images: Tensor) -> Tensor:
        """Converts images into patch tokens.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_size, image_size).

        Returns:
            Tensor with shape (batch_size, vit.seq_length-1, vit.hidden_dim) containing
            the image patch tokens.
        """
        x = self.vit.conv_proj(images)
        tokens: Tensor = x.flatten(2).transpose(1, 2)
        return tokens

    def add_prefix_tokens(self, x: Tensor, prepend_class_token: bool = True) -> Tensor:
        """Adds class token to image patch tokens.

        Args:
            x:
                Tensor with shape (batch_size, vit.seq_length-1, vit.hidden_dim)
                containing the image patch tokens
            prepend_class_token:
                Boolean flag that determines if a class token should be prepended.

        Returns:
            Tensor with shape (batch_size, vit.seq_length, vit.hidden_dim) containing
            the image patch tokens and class tokens.
        """
        if prepend_class_token:
            x = utils.prepend_class_token(x, self.vit.class_token)
        return x

    def add_pos_embed(self, x: Tensor) -> Tensor:
        """Adds positional embeddings to the input tensor based on the Vision Transformer
        (ViT) architecture in vit.

        Args:
            x:
                Input tensor with shape (batch_size, self.sequence_length, vit.hidden_dim).

        Returns:
            Tensor after adding positional embeddings, with the same shape as the input.
        """
        # TODO(Ersi:1/24) This adds positional encoding to the prefix tokens as well.
        # Give the option of not doing so, as is the case for TIMM.
        x = x + self.interpolate_pos_encoding(x)
        return x

    def _initialize_weights(self) -> None:
        # Initialize the patch embedding layer like a linear layer instead of conv
        # layer.
        w = self.vit.conv_proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize the class token.
        torch.nn.init.normal_(self.vit.class_token, std=0.02)

        # Initialize positional encoding.
        utils.initialize_2d_sine_cosine_positional_embedding(
            pos_embedding=self.vit.encoder.pos_embedding,
            has_class_token=True,
        )

        # Initialize linear layers.
        _initialize_linear_layers(self)


def _initialize_linear_layers(module: Module) -> None:
    def init(mod: Module) -> None:
        if isinstance(mod, Linear):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0)

    module.apply(init)
