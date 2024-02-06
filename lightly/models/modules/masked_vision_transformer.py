from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor


class MaskedVisionTransformer(ABC):
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

