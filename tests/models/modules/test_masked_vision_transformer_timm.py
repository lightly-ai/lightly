from typing import Optional

import pytest
import torch
from pytest_mock import MockerFixture
from torch.nn import Parameter

from lightly.utils import dependency

if not dependency.timm_vit_available():
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip("TIMM vision transformer is not available", allow_module_level=True)


from timm.models.vision_transformer import VisionTransformer

from lightly.models.modules import masked_vision_transformer_timm
from lightly.models.modules.masked_vision_transformer_timm import (
    MaskedVisionTransformerTIMM,
)

from .masked_vision_transformer_test import MaskedVisionTransformerTest


class TestMaskedVisionTransformerTIMM(MaskedVisionTransformerTest):
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
        pos_embed_initialization: str = "sincos",
    ) -> MaskedVisionTransformerTIMM:
        vit = VisionTransformer(
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads,
            embed_dim=embed_dim,
            class_token=class_token,
            reg_tokens=reg_tokens,
            global_pool="token" if class_token else "avg",
            dynamic_img_size=True,
        )
        return MaskedVisionTransformerTIMM(
            vit=vit,
            mask_token=mask_token,
            weight_initialization=weight_initialization,
            antialias=antialias,
            pos_embed_initialization=pos_embed_initialization,
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
        mock_init_weights = mocker.spy(masked_vision_transformer_timm, "init_weights")
        self.get_masked_vit(
            patch_size=32, depth=1, num_heads=1, weight_initialization=""
        )
        mock_init_weights.assert_called()

    def test__init__weight_initialization__skip(self, mocker: MockerFixture) -> None:
        mock_init_weights = mocker.spy(masked_vision_transformer_timm, "init_weights")
        self.get_masked_vit(
            patch_size=32, depth=1, num_heads=1, weight_initialization="skip"
        )
        mock_init_weights.assert_not_called()

    @pytest.mark.parametrize("antialias", [False, True])
    def test_add_pos_embed__antialias(
        self, antialias: bool, mocker: MockerFixture
    ) -> None:
        model = self.get_masked_vit(
            patch_size=32,
            depth=1,
            num_heads=1,
            antialias=antialias,
        )
        mock_resample_abs_pos_embed = mocker.spy(
            masked_vision_transformer_timm, "resample_abs_pos_embed"
        )
        x = torch.rand(2, model.sequence_length, 768)
        model.add_pos_embed(x)
        _, call_kwargs = mock_resample_abs_pos_embed.call_args
        assert call_kwargs["antialias"] == antialias
