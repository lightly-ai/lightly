from typing import Optional

import pytest
import torch
from pytest_mock import MockerFixture
from torch.nn import Parameter

from lightly.utils import dependency

if not dependency.torchvision_vit_available():
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip(
        "Torchvision vision transformer is not available", allow_module_level=True
    )


from torchvision.models import VisionTransformer

from lightly.models.modules import masked_vision_transformer_torchvision
from lightly.models.modules.masked_vision_transformer_torchvision import (
    MaskedVisionTransformerTorchvision,
)

from .masked_vision_transformer_test import MaskedVisionTransformerTest


class TestMaskedVisionTransformerTorchvision(MaskedVisionTransformerTest):
    def get_masked_vit(
        self,
        patch_size: int,
        depth: int,
        num_heads: int,
        embed_dim: int = 768,
        class_token: bool = True,
        reg_tokens: int = 0,
        mask_token: Optional[Parameter] = None,
        antialias: bool = True,
        weight_initialization: str = "",
    ) -> MaskedVisionTransformerTorchvision:
        assert class_token, "Torchvision ViT has always a class token"
        assert reg_tokens == 0, "Torchvision ViT does not support reg tokens"
        vit = VisionTransformer(
            image_size=224,
            patch_size=patch_size,
            num_layers=depth,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=embed_dim * 4,
        )
        return MaskedVisionTransformerTorchvision(
            vit=vit,
            mask_token=mask_token,
            weight_initialization=weight_initialization,
            antialias=antialias,
        )

    @pytest.mark.parametrize("mask_token", [None, Parameter(torch.rand(1, 1, 768))])
    def test__init__mask_token(self, mask_token: Optional[Parameter]) -> None:
        model = self.get_masked_vit(
            patch_size=32,
            depth=1,
            num_heads=1,
            mask_token=mask_token,
        )
        assert isinstance(model.mask_token, Parameter)

    def test__init__weight_initialization(self, mocker: MockerFixture) -> None:
        mock_initialize_linear_layers = mocker.spy(
            masked_vision_transformer_torchvision, "_initialize_linear_layers"
        )
        self.get_masked_vit(
            patch_size=32, depth=1, num_heads=1, weight_initialization=""
        )
        mock_initialize_linear_layers.assert_called()

    def test__init__weight_initialization__skip(self, mocker: MockerFixture) -> None:
        mock_initialize_linear_layers = mocker.spy(
            masked_vision_transformer_torchvision, "_initialize_linear_layers"
        )
        self.get_masked_vit(
            patch_size=32, depth=1, num_heads=1, weight_initialization="skip"
        )
        mock_initialize_linear_layers.assert_not_called()

    @pytest.mark.skip(reason="Torchvision ViT does not support forward intermediates")
    def test_forward_intermediates(
        self, device: str, mask_ratio: Optional[float], expected_sequence_length: int
    ) -> None:
        ...

    @pytest.mark.skip(reason="Torchvision ViT does not support reg tokens")
    def test_add_prefix_tokens(
        self,
        device: str,
        class_token: bool,
        reg_tokens: int,
        expected_sequence_length: int,
    ) -> None:
        ...

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize(
        "class_token,reg_tokens,expected_sequence_length",
        [
            # (False, 0, 49), # Torchvision ViT has always a class token
            (True, 0, 50),
            # (False, 2, 51), TODO(Guarin, 07/2024): Support reg_tokens > 0
            # (True, 2, 52), TODO(Guarin, 07/2024): Support reg_tokens > 0
        ],
    )
    def test_add_pos_embed(
        self,
        device: str,
        class_token: bool,
        reg_tokens: int,
        expected_sequence_length: int,
    ) -> None:
        super().test_add_pos_embed(
            device=device,
            class_token=class_token,
            reg_tokens=reg_tokens,
            expected_sequence_length=expected_sequence_length,
        )
