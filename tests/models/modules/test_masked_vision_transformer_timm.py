from typing import Optional

import pytest
import torch
from pytest_mock import MockerFixture
from torch.nn import Parameter

from lightly.models import utils
from lightly.utils import dependency

if not dependency.timm_vit_available():
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip("TIMM vision transformer is not available", allow_module_level=True)


from timm.models.vision_transformer import VisionTransformer

from lightly.models.modules import masked_vision_transformer_timm
from lightly.models.modules.masked_vision_transformer_timm import (
    MaskedVisionTransformerTIMM,
)


class TestMaskedVisionTransformerTIMM:
    @pytest.mark.parametrize("mask_token", [None, Parameter(torch.rand(1, 1, 768))])
    def test__init__mask_token(self, mask_token: Optional[Parameter]) -> None:
        vit = VisionTransformer(patch_size=32, depth=1, num_heads=1)
        model = MaskedVisionTransformerTIMM(vit=vit, mask_token=mask_token)
        assert isinstance(model.mask_token, Parameter)

    def test__init__weight_initialization(self, mocker: MockerFixture) -> None:
        vit = VisionTransformer(patch_size=32, depth=1, num_heads=1)
        mock_init_weights = mocker.spy(masked_vision_transformer_timm, "_init_weights")
        MaskedVisionTransformerTIMM(vit=vit, weight_initialization="")
        mock_init_weights.assert_called()

    def test__init__weight_initialization__skip(self, mocker: MockerFixture) -> None:
        vit = VisionTransformer(patch_size=32, depth=1, num_heads=1)
        mock_init_weights = mocker.spy(masked_vision_transformer_timm, "_init_weights")
        MaskedVisionTransformerTIMM(vit=vit, weight_initialization="skip")
        mock_init_weights.assert_not_called()

    def test_sequence_length(self) -> None:
        # TODO(Guarin, 07/2024): Support reg_tokens > 0 and test the sequence length
        # with reg_tokens > 0.
        vit = VisionTransformer(
            img_size=(224, 224),
            patch_size=32,
            class_token=True,
            reg_tokens=0,
            depth=1,
            num_heads=1,
        )
        model = MaskedVisionTransformerTIMM(vit=vit)

        # 49 for img_height//patch_size * img_width//patch_size
        # 1 for the class token
        # 0 for the reg tokens
        expected_sequence_length = 49 + 1 + 0
        assert model.sequence_length == expected_sequence_length

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("mask_ratio", [None, 0.6])
    def test_forward(self, device: str, mask_ratio: Optional[float]) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available.")

        torch.manual_seed(0)
        batch_size = 8
        embed_dim = 768
        vit = VisionTransformer(
            patch_size=32, depth=2, num_heads=2, embed_dim=embed_dim
        )
        model = MaskedVisionTransformerTIMM(vit=vit).to(device)
        images = torch.rand(
            batch_size, 3, vit.patch_embed.img_size[0], vit.patch_embed.img_size[0]
        ).to(device)

        idx_keep = None
        if mask_ratio is not None:
            idx_keep, _ = utils.random_token_mask(
                size=(batch_size, model.sequence_length),
                device=device,
                mask_ratio=mask_ratio,
            )

        class_tokens = model(images=images, idx_keep=idx_keep)

        # output shape must be correct
        assert class_tokens.shape == (batch_size, embed_dim)
        # output must have reasonable numbers
        assert torch.all(torch.not_equal(class_tokens, torch.inf))

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize(
        "mask_ratio,expected_sequence_length", [(None, 50), (0.6, 20)]
    )
    def test_encode(
        self,
        device: str,
        mask_ratio: Optional[float],
        expected_sequence_length: int,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available.")

        torch.manual_seed(0)
        batch_size = 8
        embed_dim = 768
        vit = VisionTransformer(
            patch_size=32, depth=2, num_heads=2, embed_dim=embed_dim
        )
        model = MaskedVisionTransformerTIMM(vit=vit).to(device)
        images = torch.rand(
            batch_size, 3, vit.patch_embed.img_size[0], vit.patch_embed.img_size[0]
        ).to(device)

        idx_keep = None
        if mask_ratio is not None:
            idx_keep, _ = utils.random_token_mask(
                size=(batch_size, model.sequence_length),
                device=device,
                mask_ratio=mask_ratio,
            )

        tokens = model.encode(images=images, idx_keep=idx_keep)
        assert tokens.shape == (batch_size, expected_sequence_length, embed_dim)
        assert torch.all(torch.not_equal(tokens, torch.inf))

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_images_to_tokens(self, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available.")

        torch.manual_seed(0)
        vit = VisionTransformer(patch_size=32, depth=1, num_heads=1)
        model = MaskedVisionTransformerTIMM(vit=vit)
        images = torch.rand(2, 3, 224, 224)
        assert torch.all(
            vit.patch_embed(images) == model.images_to_tokens(images=images)
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize(
        "class_token,reg_tokens,expected_sequence_length",
        [
            (False, 0, 49),
            (True, 0, 50),
            # (False, 2, 51), TODO(Guarin, 07/2024): Support reg_tokens > 0
            # (True, 2, 52), TODO(Guarin, 07/2024): Support reg_tokens > 0
        ],
    )
    def test_add_prefix_tokens(
        self,
        device: str,
        class_token: bool,
        reg_tokens: int,
        expected_sequence_length: int,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available.")

        torch.manual_seed(0)
        vit = VisionTransformer(
            patch_size=32,
            depth=1,
            num_heads=1,
            class_token=class_token,
            reg_tokens=reg_tokens,
            global_pool="token" if class_token else "avg",
        )
        model = MaskedVisionTransformerTIMM(vit=vit)
        x = torch.rand(2, 49, 768)
        assert model.add_prefix_tokens(x).shape == (2, expected_sequence_length, 768)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize(
        "class_token,reg_tokens,expected_sequence_length",
        [
            (False, 0, 49),
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
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available.")

        torch.manual_seed(0)
        vit = VisionTransformer(
            patch_size=32,
            depth=1,
            num_heads=1,
            class_token=class_token,
            reg_tokens=reg_tokens,
            global_pool="token" if class_token else "avg",
        )
        model = MaskedVisionTransformerTIMM(vit=vit)
        x = torch.rand(2, model.sequence_length, 768)
        assert model.add_pos_embed(x).shape == (2, expected_sequence_length, 768)

    @pytest.mark.parametrize("antialias", [False, True])
    def test_add_pos_embed__antialias(
        self, antialias: bool, mocker: MockerFixture
    ) -> None:
        vit = VisionTransformer(
            patch_size=32,
            depth=1,
            num_heads=1,
            dynamic_img_size=True,
        )
        model = MaskedVisionTransformerTIMM(vit=vit, antialias=antialias)
        mock_resample_abs_pos_embed = mocker.spy(
            masked_vision_transformer_timm, "resample_abs_pos_embed"
        )
        x = torch.rand(2, model.sequence_length, 768)
        model.add_pos_embed(x)
        _, call_kwargs = mock_resample_abs_pos_embed.call_args
        assert call_kwargs["antialias"] == antialias
