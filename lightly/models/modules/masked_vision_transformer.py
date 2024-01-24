from abc import ABC, abstractmethod
from typing import Optional, Union

from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer
from torch import Tensor
from torch.nn import Parameter
from torchvision.models import VisionTransformer as TorchvisionTransformer


class MaskedVisionTransformer(ABC):
    def __init__(
        self,
        vit: Union[TorchvisionTransformer, TimmVisionTransformer],
        mask_token: Union[bool, Parameter],
        device: str,
    ):
        pass

    @abstractmethod
    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
    ) -> Tensor:
        pass

    @abstractmethod
    def patch_embed(self, images: Tensor) -> Tensor:
        pass

    @abstractmethod
    def add_prefix_tokens(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def add_pos_embed(self, x: Tensor) -> Tensor:
        pass
