import pytest
import torch

from lightly.utils import dependency

if not dependency.timm_vit_available():
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip("TIMM vision transformer is not available", allow_module_level=True)

from lightly.models.modules.masked_causal_vision_transformer import (
    MaskedCausalAttention,
    MaskedCausalBlock,
    MaskedCausalVisionTransformer,
)

# MaskedCausalAttention supports only fused attention, which needs
# torch.nn.functional.scaled_dot_product_attention (PyTorch >=2.0).
skip_without_fused_attention = pytest.mark.skipif(
    not hasattr(torch.nn.functional, "scaled_dot_product_attention"),
    reason="MaskedCausalAttention supports only fused attention (PyTorch >=2.0).",
)


class TestMaskedCausalBlock:
    def test_init__forwards_only_attention_kwargs(self) -> None:
        # Block-level kwargs such as mlp_ratio must not reach the attention layer.
        block = MaskedCausalBlock(dim=24, num_heads=3, mlp_ratio=4.0, qkv_bias=True)
        assert isinstance(block.attn, MaskedCausalAttention)
        # The mlp_ratio is still applied to the block's mlp, not dropped.
        fc1 = block.mlp.fc1
        assert isinstance(fc1, torch.nn.Linear)
        assert fc1.out_features == 24 * 4


class TestMaskedCausalVisionTransformer:
    def test_init(self) -> None:
        model = MaskedCausalVisionTransformer(
            img_size=32,
            patch_size=16,
            embed_dim=24,
            depth=2,
            num_heads=3,
            mlp_ratio=4.0,
        )
        assert all(
            isinstance(block.attn, MaskedCausalAttention) for block in model.blocks
        )

    @skip_without_fused_attention
    def test_init__aim_config(self) -> None:
        # The AIM examples build the backbone without a class token and with
        # average pooling. timm asserts global_pool != "token" when class_token is
        # False, so both settings are required for the backbone to build.
        model = MaskedCausalVisionTransformer(
            img_size=32,
            patch_size=16,
            embed_dim=24,
            depth=2,
            num_heads=3,
            class_token=False,
            no_embed_class=True,
            global_pool="avg",
        )
        assert all(
            isinstance(block.attn, MaskedCausalAttention) for block in model.blocks
        )
        images = torch.rand(2, 3, 32, 32)
        sequence_length = (32 // 16) ** 2 + model.num_prefix_tokens
        mask = torch.zeros(2, sequence_length, dtype=torch.bool)
        mask[:, model.num_prefix_tokens :] = True
        features = model.forward_features(images, mask=mask)
        assert features.shape == (2, sequence_length, 24)

    @skip_without_fused_attention
    def test_forward(self) -> None:
        model = MaskedCausalVisionTransformer(
            img_size=32, patch_size=16, embed_dim=24, depth=2, num_heads=3
        )
        images = torch.rand(2, 3, 32, 32)
        features = model.forward_features(images)
        sequence_length = (32 // 16) ** 2 + model.num_prefix_tokens
        assert features.shape == (2, sequence_length, 24)

    @skip_without_fused_attention
    def test_forward__with_mask(self) -> None:
        model = MaskedCausalVisionTransformer(
            img_size=32, patch_size=16, embed_dim=24, depth=2, num_heads=3
        )
        images = torch.rand(2, 3, 32, 32)
        sequence_length = (32 // 16) ** 2 + model.num_prefix_tokens
        mask = torch.zeros(2, sequence_length, dtype=torch.bool)
        mask[:, model.num_prefix_tokens :] = True

        features = model.forward_features(images, mask=mask)
        features_no_mask = model.forward_features(images)
        assert features.shape == (2, sequence_length, 24)
        # The mask switches the patch tokens to causal attention and changes the output.
        assert not torch.allclose(features, features_no_mask)

    @skip_without_fused_attention
    def test_forward__with_mask__grad_checkpointing(self) -> None:
        torch.manual_seed(0)
        model = MaskedCausalVisionTransformer(
            img_size=32, patch_size=16, embed_dim=24, depth=2, num_heads=3
        )
        model.eval()
        images = torch.rand(2, 3, 32, 32, requires_grad=True)
        sequence_length = (32 // 16) ** 2 + model.num_prefix_tokens
        mask = torch.zeros(2, sequence_length, dtype=torch.bool)
        mask[:, model.num_prefix_tokens :] = True

        expected = model.forward_features(images, mask=mask)
        expected.sum().backward()
        assert images.grad is not None
        expected_input_grad = images.grad.detach().clone()

        model.zero_grad(set_to_none=True)
        images.grad.zero_()
        model.set_grad_checkpointing()
        features = model.forward_features(images, mask=mask)
        features.sum().backward()

        torch.testing.assert_close(features, expected)
        torch.testing.assert_close(images.grad, expected_input_grad)
