from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from torch import Tensor
from torch.nn import Module


class MaskedVisionTransformer(ABC, Module):
    """
    Abstract base class for Masked Vision Transformer models.

    Defines the interface for a Masked Vision Transformer. This class includes abstract
    methods that must be implemented by concrete subclasses to define the forward pass,
    tokenization of images, and various operations needed for the transformer.
    """

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
    ) -> Tensor:
        ...

    @abstractmethod
    def forward_intermediates(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        norm: bool = False,
    ) -> Tuple[Tensor, List[Tensor]]:
        ...

    def encode(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
    ) -> Tensor:
        ...

    @abstractmethod
    def images_to_tokens(self, images: Tensor) -> Tensor:
        ...

    @abstractmethod
    def add_prefix_tokens(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def add_pos_embed(self, x: Tensor) -> Tensor:
        ...
