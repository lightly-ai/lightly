import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, Module, Parameter
from torchvision.models.vision_transformer import VisionTransformer

from lightly.models import utils
from lightly.models.modules.masked_vision_transformer import MaskedVisionTransformer


class MaskedVisionTransformerTorchvision(MaskedVisionTransformer):
    """Masked Vision Transformer class using Torchvision.

    Attributes:
        vit:
            The VisionTransformer object of Torchvision.
        mask_token:
            The mask token.
        weight_initialization:
            The weight initialization method. Valid options are ['', 'skip']. '' uses
            the default MAE weight initialization and 'skip' skips the weight
        antialias:
            Whether to use antialiasing when resampling the positional embeddings.
        pos_embed_initialization:
            The strategy to initialize the positional embeddings. Valid options are
            ['learn', 'sincos', 'skip'].

    """

    def __init__(
        self,
        vit: VisionTransformer,
        mask_token: Optional[Parameter] = None,
        weight_initialization: str = "",
        antialias: bool = True,
        pos_embed_initialization: str = "sincos",
    ) -> None:
        super().__init__()
        self.vit = vit
        self.mask_token = (
            mask_token
            if mask_token is not None
            else Parameter(torch.zeros(1, 1, self.vit.hidden_dim))
        )
        if weight_initialization not in ("", "skip"):
            raise ValueError(
                f"Invalid weight initialization method: '{weight_initialization}'. "
                "Valid options are: ['', 'skip']."
            )
        if weight_initialization != "skip":
            self._initialize_weights()

        utils.initialize_positional_embedding(
            pos_embedding=self.vit.encoder.pos_embedding,
            strategy=pos_embed_initialization,
            num_prefix_tokens=1,  # class token
        )

        self.antialias = antialias

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
            antialias=self.antialias,
        )
        pos_embedding = pos_embedding.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embedding), dim=1)

    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        out = self.encode(images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask)
        class_token = out[:, 0]
        return class_token

    def forward_intermediates(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        norm: bool = False,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        raise NotImplementedError(
            "forward_intermediates is not implemented for this model."
        )

    def encode(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        tokens = self.preprocess(
            images=images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask
        )
        out: Tensor = self.vit.encoder.ln(
            self.vit.encoder.layers(self.vit.encoder.dropout(tokens))
        )
        return out

    def images_to_tokens(self, images: Tensor) -> Tensor:
        x = self.vit.conv_proj(images)
        tokens: Tensor = x.flatten(2).transpose(1, 2)
        return tokens

    def prepend_prefix_tokens(
        self, x: Tensor, prepend_class_token: bool = True
    ) -> Tensor:
        if prepend_class_token:
            x = utils.prepend_class_token(x, self.vit.class_token)
        return x

    def add_pos_embed(self, x: Tensor) -> Tensor:
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

        # Initialize linear layers.
        _initialize_linear_layers(self)


def _initialize_linear_layers(module: Module) -> None:
    def init(mod: Module) -> None:
        if isinstance(mod, Linear):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0)

    module.apply(init)
