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
    def test_forward_features__grad_checkpointing_matches_without_checkpointing_masked(
        self,
    ) -> None:
        torch.manual_seed(0)
        model = MaskedCausalVisionTransformer(
            img_size=32, patch_size=16, embed_dim=24, depth=2, num_heads=3
        ).eval()
        images = torch.rand(2, 3, 32, 32)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[:, 1:] = True

        expected = model.forward_features(images, mask=mask)
        model.set_grad_checkpointing(True)
        actual = model.forward_features(images, mask=mask)

        torch.testing.assert_close(actual, expected)

    @skip_without_fused_attention
    def test_forward_features__grad_checkpointing_matches_without_checkpointing_unmasked(
        self,
    ) -> None:
        torch.manual_seed(0)
        model = MaskedCausalVisionTransformer(
            img_size=32, patch_size=16, embed_dim=24, depth=2, num_heads=3
        ).eval()
        images = torch.rand(2, 3, 32, 32)

        expected = model.forward_features(images, mask=None)
        model.set_grad_checkpointing(True)
        actual = model.forward_features(images, mask=None)

        torch.testing.assert_close(actual, expected)

    @skip_without_fused_attention
    def test_forward_features__gradients_match_with_and_without_checkpointing(
        self,
    ) -> None:
        torch.manual_seed(42)
        model = MaskedCausalVisionTransformer(
            img_size=32, patch_size=16, embed_dim=24, depth=2, num_heads=3
        ).train()
        images = torch.rand(2, 3, 32, 32)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[:, 1:] = True

        # Forward and backward without checkpointing
        images_no_ckpt = images.clone().detach().requires_grad_(True)
        model.set_grad_checkpointing(False)
        out_no_ckpt = model.forward_features(images_no_ckpt, mask=mask)
        loss_no_ckpt = out_no_ckpt.sum()
        loss_no_ckpt.backward()

        grads_no_ckpt = {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        img_grad_no_ckpt = images_no_ckpt.grad.clone()

        # Reset model gradients
        model.zero_grad()

        # Forward and backward with checkpointing
        images_with_ckpt = images.clone().detach().requires_grad_(True)
        model.set_grad_checkpointing(True)
        out_with_ckpt = model.forward_features(images_with_ckpt, mask=mask)
        loss_with_ckpt = out_with_ckpt.sum()
        loss_with_ckpt.backward()

        grads_with_ckpt = {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        img_grad_with_ckpt = images_with_ckpt.grad.clone()

        # Compare outputs and gradients
        torch.testing.assert_close(out_with_ckpt, out_no_ckpt)
        torch.testing.assert_close(img_grad_with_ckpt, img_grad_no_ckpt)
        for name in grads_no_ckpt:
            torch.testing.assert_close(
                grads_with_ckpt[name],
                grads_no_ckpt[name],
                msg=f"Gradient mismatch in parameter {name}",
            )

    @skip_without_fused_attention
    @pytest.mark.parametrize("is_causal", [True, False])
    def test_forward_features__is_causal(self, is_causal: bool) -> None:
        torch.manual_seed(0)
        model = MaskedCausalVisionTransformer(
            img_size=32, patch_size=16, embed_dim=24, depth=2, num_heads=3
        ).eval()
        images = torch.rand(2, 3, 32, 32)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[:, 1:] = True

        out_no_ckpt = model.forward_features(images, mask=mask, is_causal=is_causal)
        model.set_grad_checkpointing(True)
        out_with_ckpt = model.forward_features(images, mask=mask, is_causal=is_causal)

        torch.testing.assert_close(out_with_ckpt, out_no_ckpt)
