from typing import Optional

import pytest
import torch
from pytest_mock import MockerFixture
from torch.nn import Parameter

from lightly.utils import dependency

if not dependency.timm_vit_available():
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip("TIMM vision transformer is not available", allow_module_level=True)


from timm.layers import config as timm_config
from timm.models import _manipulate
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

    @pytest.mark.parametrize("method_name", ["encode", "forward_intermediates"])
    def test_gradient_checkpointing(
        self, method_name: str, mocker: MockerFixture
    ) -> None:
        model = self.get_masked_vit(
            patch_size=32,
            depth=2,
            num_heads=1,
            embed_dim=32,
        )
        model.eval()
        images = torch.rand(2, 3, 64, 64)

        plain_images = images.clone().requires_grad_()
        expected = getattr(model, method_name)(plain_images)
        expected_output = (
            expected[0] if method_name == "forward_intermediates" else expected
        )
        expected_output.sum().backward()
        expected_param_grads = [
            param.grad.clone()
            for param in model.vit.blocks.parameters()
            if param.grad is not None
        ]

        model.zero_grad(set_to_none=True)
        model.set_grad_checkpointing(enable=True)
        spy = mocker.spy(_manipulate, "checkpoint_seq")

        checkpointed_images = images.clone().requires_grad_()
        actual = getattr(model, method_name)(checkpointed_images)
        if method_name == "forward_intermediates":
            actual_output, actual_intermediates = actual
            _, expected_intermediates = expected
            assert len(actual_intermediates) == len(expected_intermediates)
            for actual_intermediate, expected_intermediate in zip(
                actual_intermediates, expected_intermediates
            ):
                torch.testing.assert_close(actual_intermediate, expected_intermediate)
        else:
            actual_output = actual
        actual_output.sum().backward()

        expected_call_count = (
            len(model.vit.blocks) if method_name == "forward_intermediates" else 1
        )
        assert spy.call_count == expected_call_count
        torch.testing.assert_close(actual_output, expected_output)
        torch.testing.assert_close(checkpointed_images.grad, plain_images.grad)
        param_grads = [param.grad for param in model.vit.blocks.parameters()]
        assert len(param_grads) == len(expected_param_grads)
        for param_grad, expected_grad in zip(param_grads, expected_param_grads):
            torch.testing.assert_close(param_grad, expected_grad)

    @pytest.mark.parametrize("method_name", ["encode", "forward_intermediates"])
    def test_gradient_checkpointing__frozen_preprocessing(
        self, method_name: str, mocker: MockerFixture
    ) -> None:
        # timm < 1.0.16 always uses reentrant checkpointing, which drops the
        # block gradients when the checkpoint input does not require grad.
        mocker.patch.object(timm_config, "_USE_REENTRANT_CKPT", True, create=True)
        model = self.get_masked_vit(
            patch_size=32,
            depth=2,
            num_heads=1,
            embed_dim=32,
        )
        model.eval()
        for name, param in model.named_parameters():
            if not name.startswith("vit.blocks."):
                param.requires_grad_(False)
        model.set_grad_checkpointing(enable=True)

        output = getattr(model, method_name)(torch.rand(2, 3, 64, 64))
        if method_name == "forward_intermediates":
            output = output[0]
        output.sum().backward()

        for param in model.vit.blocks.parameters():
            assert param.grad is not None

    def test_gradient_checkpointing__no_grad(self) -> None:
        model = self.get_masked_vit(
            patch_size=32,
            depth=2,
            num_heads=1,
            embed_dim=32,
        )
        model.eval()
        images = torch.rand(2, 3, 64, 64)

        expected = model.encode(images)
        model.set_grad_checkpointing(enable=True)
        with torch.no_grad():
            actual = model.encode(images)

        torch.testing.assert_close(actual, expected)
