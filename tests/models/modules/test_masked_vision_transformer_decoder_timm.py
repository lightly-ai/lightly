import unittest

import pytest
import torch
from torch.nn import Linear

from lightly.models import utils
from lightly.utils import dependency

if not dependency.timm_vit_available():
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip("TIMM vision transformer is not available", allow_module_level=True)


from lightly.models.modules import (
    MAEDecoderTIMM,
    MaskedVisionTransformerDecoderTIMM,
)


class TestMaskedVisionTransformerDecoderTIMM(unittest.TestCase):
    def test_init(self) -> None:
        MaskedVisionTransformerDecoderTIMM(
            num_patches=49,
            embed_dim=256,
            depth=2,
            num_heads=4,
        )

    def _test_forward(
        self, device: torch.device, batch_size: int = 8, seed: int = 0
    ) -> None:
        torch.manual_seed(seed)
        num_patches, num_prefix_tokens, embed_dim = 49, 1, 256
        seq_length = num_patches + num_prefix_tokens
        decoder = MaskedVisionTransformerDecoderTIMM(
            num_patches=num_patches,
            embed_dim=embed_dim,
            depth=2,
            num_heads=4,
            num_prefix_tokens=num_prefix_tokens,
        ).to(device)
        tokens = torch.rand(batch_size, seq_length, embed_dim).to(device)
        out = decoder(tokens)

        # output shape must be unchanged
        self.assertListEqual(list(out.shape), [batch_size, seq_length, embed_dim])
        # output must have reasonable numbers
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_forward(self) -> None:
        self._test_forward(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available.")
    def test_forward_cuda(self) -> None:
        self._test_forward(torch.device("cuda"))

    def test_sequence_length(self) -> None:
        decoder = MaskedVisionTransformerDecoderTIMM(
            num_patches=49,
            embed_dim=256,
            depth=2,
            num_heads=4,
            num_prefix_tokens=8,
        )
        self.assertEqual(decoder.sequence_length, 49 + 8)

    def test_pos_embed__is_frozen_sine_cosine(self) -> None:
        num_patches, num_prefix_tokens, embed_dim = 49, 1, 256
        decoder = MaskedVisionTransformerDecoderTIMM(
            num_patches=num_patches,
            embed_dim=embed_dim,
            depth=2,
            num_heads=4,
            num_prefix_tokens=num_prefix_tokens,
        )
        # positional embedding has the right shape and is frozen
        self.assertListEqual(
            list(decoder.pos_embed.shape),
            [1, num_patches + num_prefix_tokens, embed_dim],
        )
        self.assertFalse(decoder.pos_embed.requires_grad)
        # prefix token positional embeddings are zero for sine-cosine embeddings
        self.assertTrue(torch.all(decoder.pos_embed[:, :num_prefix_tokens] == 0))
        # patch positional embeddings are not zero
        self.assertFalse(torch.all(decoder.pos_embed[:, num_prefix_tokens:] == 0))

    def test_preprocess__idx_mask(self) -> None:
        num_patches, embed_dim = 16, 32
        decoder = MaskedVisionTransformerDecoderTIMM(
            num_patches=num_patches,
            embed_dim=embed_dim,
            depth=1,
            num_heads=4,
            num_prefix_tokens=0,
        )
        # zero out the positional embedding to isolate the masking
        decoder.pos_embed.data.zero_()
        tokens = torch.rand(2, num_patches, embed_dim)
        idx_mask = torch.tensor([[0, 1], [2, 3]])
        out = decoder.preprocess(tokens, idx_mask=idx_mask)

        # masked positions are replaced by the mask token
        for b in range(2):
            for i in idx_mask[b].tolist():
                self.assertTrue(torch.allclose(out[b, i], decoder.mask_token[0, 0]))
        # all other positions are unchanged
        keep = [i for i in range(num_patches) if i not in {0, 1}]
        self.assertTrue(torch.allclose(out[0, keep], tokens[0, keep]))

    def test_preprocess__mask(self) -> None:
        num_patches, embed_dim = 16, 32
        decoder = MaskedVisionTransformerDecoderTIMM(
            num_patches=num_patches,
            embed_dim=embed_dim,
            depth=1,
            num_heads=4,
            num_prefix_tokens=0,
        )
        decoder.pos_embed.data.zero_()
        tokens = torch.rand(2, num_patches, embed_dim)
        mask = torch.zeros(2, num_patches, dtype=torch.bool)
        mask[:, :4] = True
        out = decoder.preprocess(tokens, mask=mask)

        # masked positions are replaced by the mask token
        self.assertTrue(
            torch.allclose(out[:, :4], decoder.mask_token.expand(2, 4, embed_dim))
        )
        # unmasked positions are unchanged
        self.assertTrue(torch.allclose(out[:, 4:], tokens[:, 4:]))

    def test_preprocess__idx_keep(self) -> None:
        num_patches, embed_dim = 16, 32
        decoder = MaskedVisionTransformerDecoderTIMM(
            num_patches=num_patches,
            embed_dim=embed_dim,
            depth=1,
            num_heads=4,
            num_prefix_tokens=0,
        )
        decoder.pos_embed.data.zero_()
        tokens = torch.rand(2, num_patches, embed_dim)
        idx_keep = torch.tensor([[0, 5, 10], [1, 2, 3]])
        out = decoder.preprocess(tokens, idx_keep=idx_keep)

        # only the kept tokens are returned
        self.assertListEqual(list(out.shape), [2, 3, embed_dim])
        for b in range(2):
            self.assertTrue(torch.allclose(out[b], tokens[b, idx_keep[b]]))

    def test_preprocess__idx_mask_and_mask_raises(self) -> None:
        decoder = MaskedVisionTransformerDecoderTIMM(
            num_patches=16,
            embed_dim=32,
            depth=1,
            num_heads=4,
            num_prefix_tokens=0,
        )
        tokens = torch.rand(2, 16, 32)
        idx_mask = torch.tensor([[0], [1]])
        mask = torch.zeros(2, 16, dtype=torch.bool)
        with self.assertRaises(ValueError):
            decoder.preprocess(tokens, idx_mask=idx_mask, mask=mask)

    def test_matches_mae_decoder_decode(self) -> None:
        # The decoder core (positional embedding, transformer blocks, norm) must be
        # equivalent to MAEDecoderTIMM.decode with the same weights.
        torch.manual_seed(0)
        num_patches, num_prefix_tokens = 49, 1
        decoder_embed_dim, depth, num_heads = 256, 2, 4
        seq_length = num_patches + num_prefix_tokens

        mae_decoder = MAEDecoderTIMM(
            num_patches=num_patches,
            patch_size=16,
            embed_dim=128,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=depth,
            decoder_num_heads=num_heads,
            num_prefix_tokens=num_prefix_tokens,
        )
        decoder = MaskedVisionTransformerDecoderTIMM(
            num_patches=num_patches,
            embed_dim=decoder_embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_prefix_tokens=num_prefix_tokens,
        )
        # copy the shared weights
        decoder.blocks.load_state_dict(mae_decoder.decoder_blocks.state_dict())
        decoder.norm.load_state_dict(mae_decoder.decoder_norm.state_dict())
        decoder.pos_embed.data.copy_(mae_decoder.decoder_pos_embed.data)

        tokens = torch.rand(4, seq_length, decoder_embed_dim)
        out_mae = mae_decoder.decode(tokens)
        out_decoder = decoder(tokens)
        self.assertTrue(torch.allclose(out_mae, out_decoder, atol=1e-6))

    def test_matches_mae_decoder_full_flow(self) -> None:
        # The full MAE flow (embed -> scatter kept tokens -> place mask tokens ->
        # decode -> predict) must be identical to MAEDecoderTIMM given the same
        # weights. This locks in that the benchmark/example migration to the new
        # decoder is behaviour-preserving.
        torch.manual_seed(0)
        num_patches, num_prefix_tokens, patch_size = 49, 1, 16
        embed_dim, decoder_embed_dim, depth, num_heads = 128, 256, 3, 4
        seq_length = num_patches + num_prefix_tokens
        out_dim = patch_size**2 * 3
        batch_size = 4

        mae_decoder = MAEDecoderTIMM(
            num_patches=num_patches,
            patch_size=patch_size,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=depth,
            decoder_num_heads=num_heads,
            num_prefix_tokens=num_prefix_tokens,
        ).eval()
        decoder = MaskedVisionTransformerDecoderTIMM(
            num_patches=num_patches,
            embed_dim=decoder_embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_prefix_tokens=num_prefix_tokens,
        ).eval()
        # embed and prediction head live outside the decoder in the new pattern
        decoder_embed = Linear(embed_dim, decoder_embed_dim)
        prediction_head = Linear(decoder_embed_dim, out_dim)
        decoder_embed.load_state_dict(mae_decoder.decoder_embed.state_dict())
        prediction_head.load_state_dict(mae_decoder.decoder_pred.state_dict())
        decoder.blocks.load_state_dict(mae_decoder.decoder_blocks.state_dict())
        decoder.norm.load_state_dict(mae_decoder.decoder_norm.state_dict())
        decoder.pos_embed.data.copy_(mae_decoder.decoder_pos_embed.data)
        decoder.mask_token.data.copy_(mae_decoder.mask_token.data)

        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, seq_length), mask_ratio=0.75
        )
        x_encoded = torch.rand(batch_size, idx_keep.shape[1], embed_dim)

        # old flow (pre-migration): fill with mask tokens, then scatter kept tokens
        x = mae_decoder.embed(x_encoded)
        x_masked = utils.repeat_token(mae_decoder.mask_token, (batch_size, seq_length))
        x_masked = utils.set_at_index(x_masked, idx_keep, x.type_as(x_masked))
        x_decoded = mae_decoder.decode(x_masked)
        expected = mae_decoder.predict(utils.get_at_index(x_decoded, idx_mask))

        # new flow: scatter kept tokens into zeros, the decoder places the mask tokens
        x = decoder_embed(x_encoded)
        x_masked = x.new_zeros(batch_size, seq_length, decoder_embed_dim)
        x_masked = utils.set_at_index(x_masked, idx_keep, x)
        x_decoded = decoder(x_masked, idx_mask=idx_mask)
        predictions = prediction_head(utils.get_at_index(x_decoded, idx_mask))

        self.assertTrue(torch.allclose(expected, predictions, atol=1e-6))
