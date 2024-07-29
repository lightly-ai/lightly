import unittest

import pytest
import torch

from lightly.models import utils
from lightly.utils import dependency

if not dependency.timm_vit_available():
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip("TIMM vision transformer is not available", allow_module_level=True)


from lightly.models.modules import IJEPAPredictorTIMM


class TestIJEPAPredictorTIMM(unittest.TestCase):
    def test_init(self) -> None:
        IJEPAPredictorTIMM(
            num_patches=196,
            depth=2,
            mlp_dim=128,
            predictor_embed_dim=128,
            num_heads=2,
            qkv_bias=False,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )

    def _test_forward(
        self, device: torch.device, batch_size: int = 4, seed: int = 0
    ) -> None:
        torch.manual_seed(seed)
        num_patches = 196  # 14x14 patches
        mlp_dim = 128
        predictor_embed_dim = 128
        depth = 3
        num_heads = 2

        predictor = IJEPAPredictorTIMM(
            num_patches=num_patches,
            depth=depth,
            mlp_dim=mlp_dim,
            predictor_embed_dim=predictor_embed_dim,
            num_heads=num_heads,
            qkv_bias=False,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        ).to(device)

        x = torch.randn(batch_size, num_patches, mlp_dim, device=device)
        masks_x = torch.randint(0, 2, (batch_size, num_patches), device=device)
        masks = torch.randint(0, 2, (batch_size, num_patches), device=device)

        predictions = predictor(x, masks_x, masks)

        # output shape must be correct
        expected_shape = [batch_size, num_patches, mlp_dim]
        self.assertListEqual(list(predictions.shape), expected_shape)

        # output must have reasonable numbers
        self.assertTrue(torch.all(torch.isfinite(predictions)))

    def test_forward(self) -> None:
        self._test_forward(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available.")
    def test_forward_cuda(self) -> None:
        self._test_forward(torch.device("cuda"))
