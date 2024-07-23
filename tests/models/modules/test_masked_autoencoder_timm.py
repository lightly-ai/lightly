import unittest

import pytest
import torch

from lightly.models import utils
from lightly.utils import dependency

if not dependency.timm_vit_available():
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip("TIMM vision transformer is not available", allow_module_level=True)


from lightly.models.modules import MAEDecoderTIMM


class TestMAEDecoderTIMM(unittest.TestCase):
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

    def _test_forward(
        self, device: torch.device, batch_size: int = 8, seed: int = 0
    ) -> None:
        torch.manual_seed(seed)
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
        self.assertListEqual(list(predictions.shape), expected_shape)

        # output must have reasonable numbers
        self.assertTrue(torch.all(torch.not_equal(predictions, torch.inf)))

    def test_forward(self) -> None:
        self._test_forward(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available.")
    def test_forward_cuda(self) -> None:
        self._test_forward(torch.device("cuda"))
