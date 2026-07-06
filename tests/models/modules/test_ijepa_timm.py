import unittest

import pytest
import torch

from lightly.models import utils
from lightly.utils import dependency

if not dependency.timm_vit_available():
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip("TIMM vision transformer is not available", allow_module_level=True)


from lightly.models.modules import IJEPAPredictorTIMM


class TestIJEPAPredictorTIMM:
    @pytest.mark.parametrize("use_stop", [True, False])
    @pytest.mark.parametrize("noise_std", [0.0, 0.1])
    def test_init(self, use_stop: bool, noise_std: float) -> None:
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
            use_stop=use_stop,
            noise_std=noise_std,
        )

    def _test_forward(
        self,
        device: torch.device,
        use_stop: bool,
        noise_std: float,
        batch_size: int = 4,
        seed: int = 0,
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
            use_stop=use_stop,
            noise_std=noise_std,
        ).to(device)

        x = torch.randn(batch_size, num_patches, mlp_dim, device=device)
        masks_x = torch.randint(0, 2, (batch_size, num_patches), device=device)
        masks = torch.randint(0, 2, (batch_size, num_patches), device=device)

        predictions = predictor(x, masks_x, masks)

        assert list(predictions.shape) == [batch_size, num_patches, mlp_dim]
        assert torch.all(torch.isfinite(predictions))

    @pytest.mark.parametrize("use_stop", [True, False])
    @pytest.mark.parametrize("noise_std", [0.0, 0.1])
    def test_forward(self, use_stop: bool, noise_std: float) -> None:
        self._test_forward(torch.device("cpu"), use_stop, noise_std)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
    @pytest.mark.parametrize("use_stop", [True, False])
    @pytest.mark.parametrize("noise_std", [0.0, 0.1])
    def test_forward_cuda(self, use_stop: bool, noise_std: float) -> None:
        self._test_forward(torch.device("cuda"), use_stop, noise_std)
