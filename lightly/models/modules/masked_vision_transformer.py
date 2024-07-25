from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from torch import Tensor
from torch.nn import Module, Parameter

from lightly.models import utils


class MaskedVisionTransformer(ABC, Module):
    """
    Abstract base class for Masked Vision Transformer models.

    Defines the interface for a Masked Vision Transformer. This class includes abstract
    methods that must be implemented by concrete subclasses to define the forward pass,
    tokenization of images, and various operations needed for the transformer.
    """

    # This is not defined as a property for backwards compatibility.
    # New models should define this as a property.
    mask_token: Parameter

    @property
    @abstractmethod
    def sequence_length(self) -> int:
        ...

    @abstractmethod
    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns encoded class tokens from a batch of images.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_size, image_size).
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
            Tensor with shape (batch_size, embed_dim) containing the encoded class token
            for every image.

        """
        ...

    @abstractmethod
    def forward_intermediates(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        norm: bool = False,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Encode input images and return features from the intermediate layers.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_height, image_width).
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                Indices must be in the range [0, sequence_length).
                If specified, the indexed tokens are masked with self.mask_token.
                Cannot be used in combination with mask argument.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, only the indexed tokens will be forwarded.
                Is applied after any masking operation.
            norm:
                Apply norm layer to all intermediates.
            mask:
                Boolean tensor with shape (batch_size, sequence_length) indicating
                which tokens should be masked. Tokens where the mask is True will be
                replaced with the mask token.
                Cannot be used in combination with idx_mask argument.

        Returns:
            Tuple of batch of encoded output tokens and a list of intermediate features.
            The encoded output tokens have shape (batch_size, embed_dim) and each
            intermediate feature has shape (batch_size, sequence_length, embed_dim).
            If idx_keep is set, only num_tokens_to_keep tokens per sequence are
            returned.
        """
        ...

    @abstractmethod
    def encode(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode input images.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_height, image_width).
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                Indices must be in the range [0, sequence_length).
                If specified, the indexed tokens are masked with self.mask_token.
                Cannot be used in combination with mask argument.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, only the indexed tokens will be encoded.
                Is applied after any masking operation.
            mask:
                Boolean tensor with shape (batch_size, sequence_length) indicating
                which tokens should be masked. Tokens where the mask is True will be
                replaced with the mask token.
                Cannot be used in combination with idx_mask argument.

        Returns:
            Tensor with shape (batch_size, sequence_length, embed_dim) containing the
            encoded output tokens. If idx_keep is set, only num_tokens_to_keep tokens
            per sequence are returned.
        """
        ...

    def preprocess(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Convert images to tokens, add positional embeddings, and apply masking.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_height, image_width).
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                Indices must be in the range [0, sequence_length).
                If specified, the indexed tokens are masked with self.mask_token.
                Cannot be used in combination with mask argument.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, only the indexed tokens will be returned.
                Is applied after any masking operation.
            mask:
                Tensor with shape (batch_size, sequence_length) indicating which tokens
                should be masked. Tokens where the mask is True will be masked with
                self.mask_token.

        Returns:
            Tensor with shape (batch_size, sequence_length, embed_dim) containing the
            preprocessed tokens. If idx_keep is set, only num_tokens_to_keep tokens
            per sequence are returned. Any class or prefix tokens are prepended to the
            sequence.
        """
        if idx_mask is not None and mask is not None:
            raise ValueError("idx_mask and mask cannot both be set at the same time.")

        # convert images to tokens
        tokens = self.images_to_tokens(images)
        # add prefix tokens if needed
        tokens = self.prepend_prefix_tokens(tokens)

        if idx_mask is not None:
            tokens = utils.mask_at_index(
                tokens=tokens, index=idx_mask, mask_token=self.mask_token
            )
        elif mask is not None:
            tokens = utils.mask_bool(
                tokens=tokens, mask=mask, mask_token=self.mask_token
            )

        # add positional encoding
        tokens = self.add_pos_embed(tokens)

        if idx_keep is not None:
            tokens = utils.get_at_index(tokens, idx_keep)

        return tokens

    @abstractmethod
    def images_to_tokens(self, images: Tensor) -> Tensor:
        """Converts images into patch tokens.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_height, image_width).

        Returns:
            Tensor with shape (batch_size, num_patches, embed_dim) containing the
            patch tokens (excluding prefix tokens).
        """
        ...

    # Keep for backwards compatibility.
    def add_prefix_tokens(self, x: Tensor) -> Tensor:
        return self.prepend_prefix_tokens(x)

    @abstractmethod
    def prepend_prefix_tokens(self, x: Tensor) -> Tensor:
        """Prepends prefix tokens to the input patch tokens.

        Args:
            x:
                Tensor with shape (batch_size, num_patches, embed_dim) containing patch
                tokens.

        Returns:
            Tensor with shape (batch_size, sequence_length, embed_dim) containing
            the prefix and patch tokens. The prefix tokens are prepended to the
            sequence.
        """
        ...

    @abstractmethod
    def add_pos_embed(self, x: Tensor) -> Tensor:
        """Adds positional embeddings to the input tokens.

        Args:
            x:
                Tensor with shape (batch_size, sequence_length, embed_dim) containing
                the input tokens. Must include prefix tokens.

        Returns:
            Tensor after adding positional embeddings, with the same shape as the input.
        """
        ...
