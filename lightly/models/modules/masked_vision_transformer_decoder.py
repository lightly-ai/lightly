from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor
from torch.nn import Module, Parameter

from lightly.models import utils


class MaskedVisionTransformerDecoder(ABC, Module):
    """Abstract base class for Masked Vision Transformer decoders.

    Defines the interface for a Masked Vision Transformer decoder (also called
    predictor). Unlike :class:`MaskedVisionTransformer`, which takes images as input,
    a decoder takes patch tokens as input. This makes it reusable for methods that
    require a ViT decoder on top of an encoder, such as MAE and I-JEPA. The input
    embedding and prediction head are intentionally kept outside of the decoder to
    make it modular and easy to reuse across methods.

    Concrete subclasses must implement the abstract methods to define the forward
    pass, the positional embedding, and the transformer blocks.
    """

    # Set by subclasses. Declared here so the shared preprocess method can
    # reference it.
    mask_token: Parameter

    @property
    @abstractmethod
    def sequence_length(self) -> int: ...

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Decodes a batch of token sequences.

        Args:
            x:
                Tensor with shape (batch_size, sequence_length, embed_dim) containing
                the input tokens.
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, the indexed tokens are masked with self.mask_token.
                Cannot be used in combination with mask argument.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, only the indexed tokens will be forwarded.
                Is applied after any masking operation.
            mask:
                Boolean tensor with shape (batch_size, sequence_length) indicating
                which tokens should be masked. Tokens where the mask is True will be
                replaced with the mask token.
                Cannot be used in combination with idx_mask argument.

        Returns:
            Tensor with shape (batch_size, sequence_length, embed_dim) containing the
            decoded output tokens. If idx_keep is set, only num_tokens_to_keep tokens
            per sequence are returned.
        """
        ...

    def preprocess(
        self,
        x: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Adds positional embeddings and applies masking to the input tokens.

        Args:
            x:
                Tensor with shape (batch_size, sequence_length, embed_dim) containing
                the input tokens.
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, the indexed tokens are masked with self.mask_token.
                Cannot be used in combination with mask argument.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, only the indexed tokens will be returned.
                Is applied after any masking operation.
            mask:
                Boolean tensor with shape (batch_size, sequence_length) indicating
                which tokens should be masked. Tokens where the mask is True will be
                replaced with the mask token.
                Cannot be used in combination with idx_mask argument.

        Returns:
            Tensor with shape (batch_size, sequence_length, embed_dim) containing the
            preprocessed tokens. If idx_keep is set, only num_tokens_to_keep tokens
            per sequence are returned.
        """
        if idx_mask is not None and mask is not None:
            raise ValueError("idx_mask and mask cannot both be set at the same time.")

        if idx_mask is not None:
            x = utils.mask_at_index(
                tokens=x, index=idx_mask, mask_token=self.mask_token
            )
        elif mask is not None:
            x = utils.mask_bool(tokens=x, mask=mask, mask_token=self.mask_token)

        # add positional encoding
        x = self.add_pos_embed(x)

        if idx_keep is not None:
            x = utils.get_at_index(x, idx_keep)

        return x

    @abstractmethod
    def add_pos_embed(self, x: Tensor) -> Tensor:
        """Adds positional embeddings to the input tokens.

        Args:
            x:
                Tensor with shape (batch_size, sequence_length, embed_dim) containing
                the input tokens.

        Returns:
            Tensor after adding positional embeddings, with the same shape as the input.
        """
        ...

    @abstractmethod
    def decode(self, x: Tensor) -> Tensor:
        """Forwards the tokens through the transformer blocks and the final norm layer.

        Args:
            x:
                Tensor with shape (batch_size, sequence_length, embed_dim) containing
                the input tokens.

        Returns:
            Tensor with shape (batch_size, sequence_length, embed_dim) containing the
            decoded tokens.
        """
        ...
