from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor


class MaskedVisionTransformer(ABC):
    """
    Abstract base class for Masked Vision Transformer models.

    Defines the interface for a Masked Vision Transformer. This class includes abstract
    methods that must be implemented by concrete subclasses to define the forward pass,
    tokenization of images, and various operations needed for the transformer.
    """

    @abstractmethod
    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
    ) -> Tensor:
        pass

    @abstractmethod
    def images_to_tokens(self, images: Tensor) -> Tensor:
        pass

    @abstractmethod
    def add_prefix_tokens(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def add_pos_embed(self, x: Tensor) -> Tensor:
        pass
