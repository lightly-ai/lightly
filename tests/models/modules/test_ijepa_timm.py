import warnings
from functools import partial

import pytest
import torch
from torch import Tensor, nn

from lightly.models import utils
from lightly.utils import dependency

if not dependency.timm_vit_available():
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip("TIMM vision transformer is not available", allow_module_level=True)


from timm.models.vision_transformer import Block

from lightly.models.modules import IJEPAPredictorTIMM


class _LegacyIJEPAPredictorTIMM(nn.Module):
    """Pre-refactor I-JEPA predictor, kept for checkpoint migration tests.

    Mirrors the standalone implementation from before the transformer core was
    factored out into the decoder submodule, so the migration hook can be checked
    against the exact old parameter names and forward pass.
    """

    def __init__(
        self,
        num_patches: int,
        depth: int,
        mlp_dim: int,
        predictor_embed_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.predictor_embed = nn.Linear(mlp_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.predictor_proj = nn.Linear(predictor_embed_dim, mlp_dim, bias=True)
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
        )
        pos_embed = utils.get_2d_sincos_pos_embed(
            self.predictor_pos_embed.shape[-1], int(num_patches**0.5), cls_token=False
        )
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop_path=0.0,
                    proj_drop=0.0,
                    attn_drop=0.0,
                    norm_layer=norm_layer,  # type: ignore[arg-type]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor, masks_x: Tensor, masks: Tensor) -> Tensor:
        batch_size = len(x)
        x = self.predictor_embed(x)
        x_pos_embed = self.predictor_pos_embed.repeat(batch_size, 1, 1)
        x += utils.apply_masks(x_pos_embed, masks_x)
        _, num_context, _ = x.shape
        pos_embs = self.predictor_pos_embed.repeat(batch_size, 1, 1)
        pos_embs = utils.apply_masks(pos_embs, masks)
        pos_embs = utils.repeat_interleave_batch(pos_embs, batch_size, repeat=1)
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs
        x = torch.cat([x, pred_tokens], dim=1)
        for block in self.predictor_blocks:
            x = block(x)
        x = self.predictor_norm(x)
        x = x[:, num_context:]
        x = self.predictor_proj(x)
        return x


class TestIJEPAPredictorTIMM:
    @pytest.mark.parametrize("noise_std", [0.0, 0.1])
    def test_init(self, noise_std: float) -> None:
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
            noise_std=noise_std,
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_forward(self, device: str) -> None:
        if not torch.cuda.is_available() and device == "cuda":
            pytest.skip("CUDA not available")

        torch.manual_seed(0)
        batch_size = 4
        num_patches = 196  # 14x14 patches
        mlp_dim = 128
        predictor_embed_dim = 128
        depth = 3
        num_heads = 2
        noise_std = 0.1

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
            noise_std=noise_std,
        ).to(device)

        x = torch.randn(batch_size, num_patches, mlp_dim, device=device)
        masks_x = torch.randint(0, 2, (batch_size, num_patches), device=device)
        masks = torch.randint(0, 2, (batch_size, num_patches), device=device)

        predictions = predictor(x, masks_x, masks)

        # output shape must be correct
        expected_shape = [batch_size, num_patches, mlp_dim]
        assert list(predictions.shape) == expected_shape

        # output must have reasonable numbers
        assert torch.all(torch.isfinite(predictions))

    def test_migrates_legacy_checkpoint(self) -> None:
        # A checkpoint saved with the pre-refactor parameter names must still load,
        # emit a deprecation warning, and reproduce the old predictor's output.
        torch.manual_seed(0)
        legacy = _LegacyIJEPAPredictorTIMM(
            num_patches=196, depth=2, mlp_dim=128, predictor_embed_dim=128, num_heads=2
        ).eval()
        predictor = IJEPAPredictorTIMM(
            num_patches=196, depth=2, mlp_dim=128, predictor_embed_dim=128, num_heads=2
        ).eval()

        # Loading the old-named keys triggers the migration hook and warns.
        with pytest.warns(DeprecationWarning):
            predictor.load_state_dict(legacy.state_dict())

        x = torch.randn(4, 196, 128)
        masks_x = torch.randint(0, 2, (4, 196))
        masks = torch.randint(0, 2, (4, 196))
        assert torch.allclose(
            legacy(x, masks_x, masks), predictor(x, masks_x, masks), atol=1e-6
        )

    def test_current_checkpoint_loads_without_migration_warning(self) -> None:
        # A checkpoint saved with the current names must not trigger the migration.
        predictor = IJEPAPredictorTIMM(
            num_patches=196, depth=2, mlp_dim=128, predictor_embed_dim=128, num_heads=2
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predictor.load_state_dict(predictor.state_dict())
        assert not any(
            "predictor_pos_embed" in str(warning.message) for warning in caught
        )
