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
    def sequence_length(self) -> int: ...

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
                If specified, the indexed tokens are masked with self.mask_token.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                If specified, only the indexed tokens will be passed to the
                encoder.
            mask:
                Tensor with shape (batch_size, sequence_length) indicating which tokens
                should be masked. Tokens where the mask is True will be replaced with
                the mask token.

        Returns:
            Tensor with shape (batch_size, embed_dim) containing the
            encoded class token for every image.

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
                Batch of input images.
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                If specified, the indexed tokens are masked with self.mask_token.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                If specified, only the indexed tokens will be encoded.
            norm:
                Apply norm layer to all intermediates.
            mask:
                Tensor with shape (batch_size, sequence_length) indicating which tokens
                should be masked. Tokens where the mask is True will be replaced with
                the mask token.

        Returns:
            Tuple of batch of encoded output tokens and a list of intermediate features
            from each layer with shape (batch_size, sequence_length, embed_dim).
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
                Batch of input images.
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                If specified, the indexed tokens are masked with self.mask_token.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                If specified, only the indexed tokens will be encoded.
            mask:
                Tensor with shape (batch_size, sequence_length) indicating which tokens
                should be masked. Tokens where the mask is True will be replaced with
                the mask token.

        Returns:
            Batch of encoded output tokens.
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
                Batch of input images.
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                If specified, the indexed tokens are masked with self.mask_token.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                If specified, only the indexed tokens will be encoded.
            mask:
                Tensor with shape (batch_size, sequence_length) indicating which tokens
                should be masked. Tokens where the mask is True will be masked with
                self.mask_token.

        Returns:
            Batch of preprocessed tokens.
        """
        if idx_mask is not None and mask is not None:
            raise ValueError("idx_mask and mask cannot both be set at the same time.")

        # convert images to tokens
        tokens = self.images_to_tokens(images)
        # add prefix tokens if needed
        tokens = self.add_prefix_tokens(tokens)

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
                Tensor with shape (batch_size, channels, image_size, image_size).

        Returns:
            Tensor with shape (batch_size, num_patches, embed_dim)
            containing the patch tokens (excluding prefix tokens).
        """
        ...

    @abstractmethod
    def add_prefix_tokens(self, x: Tensor) -> Tensor:
        """Adds prefix tokens to image patch tokens.

        Args:
            x:
                Tensor with shape (batch_size, num_patches, embed_dim)
                containing the image patch tokens

        Returns:
            Tensor with shape (batch_size, sequence_length, embed_dim) containing
            the image patch tokens and prefix tokens.
        """
        ...

    @abstractmethod
    def add_pos_embed(self, x: Tensor) -> Tensor:
        """Adds positional embeddings to the input tensor based on the Vision Transformer
        (ViT) architecture in vit.

        Args:
            x:
                Input tensor with shape (batch_size, sequence_length, embed_dim).

        Returns:
            Tensor after adding positional embeddings, with the same shape as the input.
        """
        ...
