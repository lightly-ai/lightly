from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor
from torch.nn import Parameter


class MaskedVisionTransformer(ABC):
    def __init__(
        self,
        mask_token: Optional[Parameter] = None,
    ):
        self.mask_token = mask_token

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
    def add_prefix_tokens(self, x: Tensor, prepend_class_token: bool = True) -> Tensor:
        pass

    @abstractmethod
    def add_pos_embed(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def _initialize_weights(self) -> None:
        pass
