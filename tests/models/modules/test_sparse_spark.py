import pytest
import torch
from torch import Tensor

pytest.importorskip("timm.models.layers")

import lightly.models.modules.sparse_spark as sparse_spark
from lightly.models.modules.sparse_spark import (
    SparKDensifiyBlock,
    SparKMasker,
    SparKMaskingOutput,
)


def _create_mask() -> Tensor:
    return torch.tensor(
        [
            [
                [
                    [1, 0],
                    [0, 1],
                ]
            ]
        ],
        dtype=torch.bool,
    )


def test__get_active_ex_or_ii_expands_mask() -> None:
    H, W = 32, 32
    with sparse_spark.sparse_layer_context(_create_mask()):
        active = sparse_spark._get_active_ex_or_ii(H=H, W=W, returning_active_ex=True)
        assert not isinstance(active, tuple)

        assert active.shape == (1, 1, H, W)
        assert active[:, :, :16, :16].all()
        assert active[:, :, :16, 16:].logical_not().all()
        assert active[:, :, 16:, :16].logical_not().all()
        assert active[:, :, 16:, 16:].all()


def test__get_active_ex_or_ii_dont_shrink_mask() -> None:
    H, W = 4, 4
    with sparse_spark.sparse_layer_context(torch.ones(1, 1, 32, 32)):
        with pytest.raises(AssertionError):
            sparse_spark._get_active_ex_or_ii(H=H, W=W, returning_active_ex=False)


def test__get_active_ex_or_ii_raise_on_non_active_mask() -> None:
    H, W = 32, 32
    with pytest.raises(RuntimeError):
        sparse_spark._get_active_ex_or_ii(H=H, W=W, returning_active_ex=False)


def test__get_active_ex_or_ii_returning_ex_false_correct_values() -> None:
    H, W = 32, 32
    with sparse_spark.sparse_layer_context(_create_mask()):
        active_b, active_h, active_w = sparse_spark._get_active_ex_or_ii(
            H=H, W=W, returning_active_ex=False
        )
        active_ex = sparse_spark._get_active_ex_or_ii(
            H=H, W=W, returning_active_ex=True
        )
        assert not isinstance(active_ex, tuple)

        active_ex_scattered = torch.zeros_like(active_ex)
        active_ex_scattered[active_b, :, active_h, active_w] = 1

        assert torch.equal(active_ex, active_ex_scattered)


def test__sp_conv_forward() -> None:
    H, W = 32, 32
    with sparse_spark.sparse_layer_context(_create_mask()):
        conv = sparse_spark.SparseConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )
        conv.weight.data.fill_(1)
        assert conv.bias is not None
        conv.bias.data.fill_(0)

        x = torch.ones(1, 1, H, W)
        out = conv(x)

        assert out.shape == (1, 1, H, W)
        assert out[:, :, :16, :16].all()
        assert out[:, :, :16, 16:].logical_not().all()
        assert out[:, :, 16:, :16].logical_not().all()
        assert out[:, :, 16:, 16:].all()


class TestSparKMasker:
    def test_forward(self) -> None:
        masker = SparKMasker(
            feature_map_size=(4, 4),
            downsample_ratio=8,
        )
        x = torch.ones(1, 1, 32, 32)
        mask: SparKMaskingOutput = masker(x)

        for i in range(len(mask.per_level_mask)):
            mask_current = mask.per_level_mask[i]

            assert mask_current.shape[0] == 1
            assert mask_current.shape[1] == 1
            assert mask_current.shape[2] == 4 * (2**i)
            assert mask_current.shape[3] == 4 * (2**i)

        for i in range(len(mask.per_level_mask) - 1):
            mask_current = mask.per_level_mask[i]
            mask_next = mask.per_level_mask[i + 1]
            assert mask_current.shape[2] * 2 == mask_next.shape[2]
            assert mask_current.shape[3] * 2 == mask_next.shape[3]
            assert (
                mask_next
                == mask_current.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
            ).all()

    def test_masked_bchw_applies_mask(self) -> None:
        masker = SparKMasker(
            feature_map_size=(4, 4),
            downsample_ratio=8,
        )
        x = torch.arange(1, 1025, dtype=torch.float32).view(1, 1, 32, 32)
        mask: SparKMaskingOutput = masker(x)

        # masked_bchw should be x * active_mask at full resolution
        active_mask_full = mask.per_level_mask[-1]
        expected = x * active_mask_full
        assert torch.equal(mask.masked_bchw, expected)

    def test_mask_ratio_zero_all_active(self) -> None:
        masker = SparKMasker(
            feature_map_size=(4, 4),
            downsample_ratio=8,
            mask_ratio=0.0,
        )
        x = torch.ones(1, 1, 32, 32)
        mask: SparKMaskingOutput = masker(x)

        # All tokens should be active (True)
        assert mask.per_level_mask[0].all()

    def test_batch_independence(self) -> None:
        torch.manual_seed(42)  # Deterministic masks
        masker = SparKMasker(
            feature_map_size=(4, 4),
            downsample_ratio=8,
        )
        x = torch.ones(4, 1, 32, 32)
        mask: SparKMaskingOutput = masker(x)

        # Each batch sample should have independent mask
        assert mask.per_level_mask[0].shape[0] == 4
        # Masks should differ across batch (with seed, guaranteed different)
        assert not torch.equal(mask.per_level_mask[0][0], mask.per_level_mask[0][1])


class TestSparKDensifiyBlock:
    def test_fill_with_mask_tokens_preserves_active_regions(self) -> None:
        block: SparKDensifiyBlock = SparKDensifiyBlock(
            e_width=4, d_width=4, densify_norm_str="none"
        )
        block.mask_token.data.fill_(99.0)

        features: Tensor = torch.arange(16, dtype=torch.float32).view(1, 4, 2, 2)
        active_mask: Tensor = torch.tensor([[[[True, False], [False, True]]]])

        result: Tensor = block._fill_with_mask_tokens(features, active_mask)

        assert result[0, :, 0, 0].equal(features[0, :, 0, 0])
        assert result[0, :, 1, 1].equal(features[0, :, 1, 1])

    def test_fill_with_mask_tokens_fills_inactive_regions(self) -> None:
        block: SparKDensifiyBlock = SparKDensifiyBlock(
            e_width=4, d_width=4, densify_norm_str="none"
        )
        block.mask_token.data.fill_(99.0)

        features: Tensor = torch.arange(16, dtype=torch.float32).view(1, 4, 2, 2)
        active_mask: Tensor = torch.tensor([[[[True, False], [False, True]]]])

        result: Tensor = block._fill_with_mask_tokens(features, active_mask)

        expected_token: Tensor = torch.full((4,), 99.0, dtype=torch.float32)
        assert result[0, :, 0, 1].equal(expected_token)
        assert result[0, :, 1, 0].equal(expected_token)

    def test_fill_with_mask_tokens_all_active(self) -> None:
        block: SparKDensifiyBlock = SparKDensifiyBlock(
            e_width=4, d_width=4, densify_norm_str="none"
        )
        block.mask_token.data.fill_(99.0)

        features: Tensor = torch.arange(16, dtype=torch.float32).view(1, 4, 2, 2)
        active_mask: Tensor = torch.ones(1, 1, 2, 2, dtype=torch.bool)

        result: Tensor = block._fill_with_mask_tokens(features, active_mask)

        assert result.equal(features)

    def test_fill_with_mask_tokens_all_inactive(self) -> None:
        block: SparKDensifiyBlock = SparKDensifiyBlock(
            e_width=4, d_width=4, densify_norm_str="none"
        )
        block.mask_token.data.fill_(99.0)

        features: Tensor = torch.arange(16, dtype=torch.float32).view(1, 4, 2, 2)
        active_mask: Tensor = torch.zeros(1, 1, 2, 2, dtype=torch.bool)

        result: Tensor = block._fill_with_mask_tokens(features, active_mask)

        expected: Tensor = torch.full_like(features, 99.0)
        assert result.equal(expected)

    def test_fill_with_mask_tokens_batch_independence(self) -> None:
        block: SparKDensifiyBlock = SparKDensifiyBlock(
            e_width=4, d_width=4, densify_norm_str="none"
        )
        block.mask_token.data.fill_(99.0)

        features: Tensor = torch.arange(32, dtype=torch.float32).view(2, 4, 2, 2)
        active_mask: Tensor = torch.tensor(
            [
                [[[True, False], [False, True]]],
                [[[False, True], [True, False]]],
            ]
        )

        result: Tensor = block._fill_with_mask_tokens(features, active_mask)

        expected_token: Tensor = torch.full((4,), 99.0, dtype=torch.float32)
        # Batch 0: (0,0) active, (0,1) inactive
        assert result[0, :, 0, 0].equal(features[0, :, 0, 0])
        assert result[0, :, 0, 1].equal(expected_token)
        # Batch 1: (0,0) inactive, (0,1) active
        assert result[1, :, 0, 0].equal(expected_token)
        assert result[1, :, 0, 1].equal(features[1, :, 0, 1])


class TestSparKDensifier:
    def test_forward_raises_without_active_mask(self) -> None:
        """Test that forward raises RuntimeError when not in sparse_layer_context."""
        densifier: sparse_spark.SparKDensifier = sparse_spark.SparKDensifier(
            encoder_in_channels=[64, 128, 256],
            decoder_in_channel=256,
            densify_norm_str="bn",
            sbn=False,
        )
        # Create dummy feature maps (shallow to deep)
        fea_bcffs: list[Tensor] = [
            torch.randn(1, 64, 8, 8),
            torch.randn(1, 128, 4, 4),
            torch.randn(1, 256, 2, 2),
        ]

        # Should raise because _cur_active is not set
        with pytest.raises(RuntimeError, match="_cur_active must be set"):
            densifier(fea_bcffs)

    def test_forward_upsamples_active_mask(self) -> None:
        """Test that active_fmap_current is upsampled by 2x at each iteration."""
        densifier: sparse_spark.SparKDensifier = sparse_spark.SparKDensifier(
            encoder_in_channels=[64, 128, 256],
            decoder_in_channel=256,
            densify_norm_str="bn",
            sbn=False,
        )
        # Create dummy feature maps at different scales (shallow to deep)
        fea_bcffs: list[Tensor] = [
            torch.randn(1, 64, 8, 8),
            torch.randn(1, 128, 4, 4),
            torch.randn(1, 256, 2, 2),
        ]
        # Initial active mask at deepest scale (2x2)
        initial_mask: Tensor = torch.tensor(
            [[[[True, False], [False, True]]]], dtype=torch.bool
        )

        with sparse_spark.sparse_layer_context(initial_mask):
            to_dec: list[Tensor] = densifier(fea_bcffs)

        # Check that we have 3 output feature maps (one per block)
        assert len(to_dec) == 3

        # Verify upsampling: each level should double in spatial size
        # Level 0 (deepest): 2x2 -> input to first block
        # Level 1: 2x2 -> 4x4
        # Level 2: 4x4 -> 8x8
        assert to_dec[0].shape == (1, 256, 2, 2)  # First block output
        assert to_dec[1].shape == (1, 128, 4, 4)  # Second block output
        assert to_dec[2].shape == (1, 64, 8, 8)  # Third block output

    def test_forward_to_dec_contains_all_feature_maps(self) -> None:
        """Test that to_dec list contains all densified feature maps."""
        encoder_in_channels: list[int] = [64, 128, 256]
        decoder_in_channel: int = 256
        densifier: sparse_spark.SparKDensifier = sparse_spark.SparKDensifier(
            encoder_in_channels=encoder_in_channels,
            decoder_in_channel=decoder_in_channel,
            densify_norm_str="bn",
            sbn=False,
        )
        fea_bcffs: list[Tensor] = [
            torch.randn(1, 64, 8, 8),
            torch.randn(1, 128, 4, 4),
            torch.randn(1, 256, 2, 2),
        ]
        initial_mask: Tensor = torch.ones(1, 1, 2, 2, dtype=torch.bool)

        with sparse_spark.sparse_layer_context(initial_mask):
            to_dec: list[Tensor] = densifier(fea_bcffs)

        # Should have one output per block
        assert len(to_dec) == len(densifier.blocks)
        # All outputs should be Tensors
        for i, td in enumerate(to_dec):
            assert isinstance(td, Tensor)

        # Verify channel progression: [256, 128, 64] (halved at each level after first)
        # Block 0: e_width=256 -> d_width=256 (identity)
        # Block 1: e_width=128 -> d_width=128 (256//2)
        # Block 2: e_width=64 -> d_width=64 (128//2)
        expected_channels: list[int] = [256, 128, 64]
        for i, td in enumerate(to_dec):
            assert (
                td.shape[1] == expected_channels[i]
            ), f"Block {i}: expected {expected_channels[i]} channels, got {td.shape[1]}"

    def test_forward_reverses_fea_bcffs_order(self) -> None:
        """Test that fea_bcffs is reversed before processing (deepest first)."""
        encoder_in_channels: list[int] = [64, 128, 256]
        densifier: sparse_spark.SparKDensifier = sparse_spark.SparKDensifier(
            encoder_in_channels=encoder_in_channels,
            decoder_in_channel=256,
            densify_norm_str="bn",
            sbn=False,
        )
        # Use distinct shapes to verify order
        fea_bcffs: list[Tensor] = [
            torch.randn(1, 64, 8, 8),  # shallowest
            torch.randn(1, 128, 4, 4),
            torch.randn(1, 256, 2, 2),  # deepest
        ]
        initial_mask: Tensor = torch.ones(1, 1, 2, 2, dtype=torch.bool)

        with sparse_spark.sparse_layer_context(initial_mask):
            to_dec: list[Tensor] = densifier(fea_bcffs)

        # First block should process deepest feature (2x2, 256 channels)
        assert to_dec[0].shape == (1, 256, 2, 2)
        # Second block should process middle feature (4x4, 128 channels)
        assert to_dec[1].shape == (1, 128, 4, 4)
        # Third block should process shallowest feature (8x8, 64 channels)
        assert to_dec[2].shape == (1, 64, 8, 8)

    def test_forward_complete_integration(self) -> None:
        """Integration test for complete forward pass with proper context."""
        encoder_in_channels: list[int] = [64, 128, 256]
        decoder_in_channel: int = 256
        densifier: sparse_spark.SparKDensifier = sparse_spark.SparKDensifier(
            encoder_in_channels=encoder_in_channels,
            decoder_in_channel=decoder_in_channel,
            densify_norm_str="bn",
            sbn=False,
        )
        batch_size: int = 2
        fea_bcffs: list[Tensor] = [
            torch.randn(batch_size, 64, 8, 8),
            torch.randn(batch_size, 128, 4, 4),
            torch.randn(batch_size, 256, 2, 2),
        ]
        # Active mask at deepest level (matches feature map size)
        initial_mask: Tensor = torch.randint(
            0, 2, (batch_size, 1, 2, 2), dtype=torch.bool
        )

        with sparse_spark.sparse_layer_context(initial_mask):
            to_dec: list[Tensor] = densifier(fea_bcffs)

        # Verify output structure
        assert len(to_dec) == 3
        assert all(isinstance(t, Tensor) for t in to_dec)
        # Verify batch size preserved
        assert all(t.shape[0] == batch_size for t in to_dec)
        # Verify spatial dimensions double at each level
        assert to_dec[0].shape[2] * 2 == to_dec[1].shape[2]
        assert to_dec[1].shape[2] * 2 == to_dec[2].shape[2]

    def test_forward_with_identity_proj_first_block(self) -> None:
        """Test that first block uses identity projection when channels match."""
        encoder_in_channels: list[int] = [256, 128, 64]  # Same as decoder
        densifier: sparse_spark.SparKDensifier = sparse_spark.SparKDensifier(
            encoder_in_channels=encoder_in_channels,
            decoder_in_channel=256,
            densify_norm_str="bn",
            sbn=False,
        )
        fea_bcffs: list[Tensor] = [
            torch.randn(1, 256, 8, 8),
            torch.randn(1, 128, 4, 4),
            torch.randn(1, 64, 2, 2),
        ]
        initial_mask: Tensor = torch.ones(1, 1, 2, 2, dtype=torch.bool)

        with sparse_spark.sparse_layer_context(initial_mask):
            to_dec: list[Tensor] = densifier(fea_bcffs)

        # First block should preserve channels (identity projection)
        assert to_dec[0].shape[1] == 256
