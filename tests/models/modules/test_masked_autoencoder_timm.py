import pytest
import torch

from lightly.models import utils
from lightly.utils import dependency

if not dependency.timm_vit_available():
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip("TIMM vision transformer is not available", allow_module_level=True)


from lightly.models.modules import MAEDecoderTIMM, PixioDecoderTIMM


@pytest.mark.filterwarnings("ignore::FutureWarning")
class TestMAEDecoderTIMM:
    def test_init__is_deprecated(self) -> None:
        with pytest.warns(FutureWarning):
            MAEDecoderTIMM(
                num_patches=49,
                patch_size=32,
                embed_dim=128,
                decoder_embed_dim=256,
                decoder_depth=2,
                decoder_num_heads=4,
            )

    def test_init(self) -> None:
        MAEDecoderTIMM(
            num_patches=49,
            patch_size=32,
            embed_dim=128,
            decoder_embed_dim=256,
            decoder_depth=2,
            decoder_num_heads=4,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_forward(self, device: str) -> None:
        if not torch.cuda.is_available() and device == "cuda":
            pytest.skip("CUDA not available")

        torch.manual_seed(0)
        batch_size = 8
        seq_length = 50
        embed_input_dim = 128
        patch_size = 32
        out_dim = 3 * patch_size**2
        decoder = MAEDecoderTIMM(
            num_patches=49,
            patch_size=32,
            embed_dim=embed_input_dim,
            decoder_embed_dim=256,
            decoder_depth=2,
            decoder_num_heads=4,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        ).to(device)
        tokens = torch.rand(batch_size, seq_length, embed_input_dim).to(device)
        predictions = decoder(tokens)

        # output shape must be correct
        expected_shape = [batch_size, seq_length, out_dim]
        assert list(predictions.shape) == expected_shape

        # output must have reasonable numbers
        assert torch.all(torch.not_equal(predictions, torch.inf))

    def test_init__num_prefix_tokens(self) -> None:
        num_patches, num_prefix_tokens, decoder_embed_dim = 49, 8, 256
        decoder = MAEDecoderTIMM(
            num_patches=num_patches,
            patch_size=32,
            embed_dim=128,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=2,
            decoder_num_heads=4,
            num_prefix_tokens=num_prefix_tokens,
        )
        assert list(decoder.decoder_pos_embed.shape) == [
            1,
            num_patches + num_prefix_tokens,
            decoder_embed_dim,
        ]

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_forward__num_prefix_tokens(self, device: str) -> None:
        if not torch.cuda.is_available() and device == "cuda":
            pytest.skip("CUDA not available")

        torch.manual_seed(0)
        num_patches, num_prefix_tokens, embed_input_dim, patch_size = 49, 8, 128, 32
        seq_length = num_patches + num_prefix_tokens
        decoder = MAEDecoderTIMM(
            num_patches=num_patches,
            patch_size=patch_size,
            embed_dim=embed_input_dim,
            decoder_embed_dim=256,
            decoder_depth=2,
            decoder_num_heads=4,
            num_prefix_tokens=num_prefix_tokens,
        ).to(device)
        tokens = torch.rand(2, seq_length, embed_input_dim).to(device)
        predictions = decoder(tokens)
        assert list(predictions.shape) == [2, seq_length, 3 * patch_size**2]
        assert torch.all(torch.not_equal(predictions, torch.inf))


@pytest.mark.filterwarnings("ignore::FutureWarning")
class TestPixioDecoderTIMM:
    def test_init__is_deprecated(self) -> None:
        with pytest.warns(FutureWarning):
            PixioDecoderTIMM(
                num_patches=64,
                patch_size=16,
                embed_dim=128,
                decoder_embed_dim=64,
                decoder_depth=2,
                decoder_num_heads=4,
                num_prefix_tokens=8,
            )

    def test_init__default_depth_is_32(self) -> None:
        decoder = PixioDecoderTIMM(
            num_patches=256,
            patch_size=16,
            embed_dim=768,
            decoder_embed_dim=512,
            decoder_num_heads=16,
            num_prefix_tokens=8,
        )
        assert len(decoder.decoder_blocks) == 32
        assert list(decoder.decoder_pos_embed.shape) == [1, 256 + 8, 512]

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_forward(self, device: str) -> None:
        if not torch.cuda.is_available() and device == "cuda":
            pytest.skip("CUDA not available")

        torch.manual_seed(0)
        num_patches, num_prefix_tokens, patch_size = 64, 8, 16
        seq_length = num_patches + num_prefix_tokens
        decoder = PixioDecoderTIMM(
            num_patches=num_patches,
            patch_size=patch_size,
            embed_dim=128,
            decoder_embed_dim=64,
            decoder_depth=2,  # keep the test cheap
            decoder_num_heads=4,
            num_prefix_tokens=num_prefix_tokens,
        ).to(device)
        tokens = torch.rand(2, seq_length, 128).to(device)
        predictions = decoder(tokens)
        assert list(predictions.shape) == [2, seq_length, 3 * patch_size**2]
        assert torch.all(torch.not_equal(predictions, torch.inf))
