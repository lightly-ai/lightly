from __future__ import annotations

import pytest
import torch

from lightly.utils import dependency

if not dependency.timm_vit_available():
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip("TIMM vision transformer is not available", allow_module_level=True)


from lightly.models.modules import CAPIPredictorTIMM


class TestCAPIPredictorTIMM:
    def test_forward__shape(self) -> None:
        embed_dim, grid_size = 24, 4  # 4x4 = 16 patch positions
        predictor = CAPIPredictorTIMM(
            embed_dim=embed_dim, grid_size=grid_size, depth=2, num_heads=3
        )
        batch_size, num_context, num_queries = 2, 10, 6
        context = torch.randn(batch_size, num_context, embed_dim)
        context_positions = torch.randint(0, grid_size**2, (batch_size, num_context))
        query_positions = torch.randint(0, grid_size**2, (batch_size, num_queries))
        out = predictor(
            context=context,
            context_positions=context_positions,
            query_positions=query_positions,
        )
        assert out.shape == (batch_size, num_queries, embed_dim)

    def test_forward__is_differentiable(self) -> None:
        predictor = CAPIPredictorTIMM(embed_dim=24, grid_size=4, depth=1, num_heads=3)
        context = torch.randn(2, 8, 24, requires_grad=True)
        context_positions = torch.randint(0, 16, (2, 8))
        query_positions = torch.randint(0, 16, (2, 5))
        out = predictor(
            context=context,
            context_positions=context_positions,
            query_positions=query_positions,
        )
        out.sum().backward()
        assert context.grad is not None
        assert predictor.mask_token.grad is not None

    def test_forward__is_position_dependent(self) -> None:
        # Rotary embeddings make the prediction depend on the query grid position.
        torch.manual_seed(0)
        predictor = CAPIPredictorTIMM(embed_dim=24, grid_size=4, depth=2, num_heads=3)
        context = torch.randn(1, 8, 24)
        context_positions = torch.arange(8).unsqueeze(0)
        out_a = predictor(
            context=context,
            context_positions=context_positions,
            query_positions=torch.tensor([[0, 1]]),
        )
        out_b = predictor(
            context=context,
            context_positions=context_positions,
            query_positions=torch.tensor([[10, 11]]),
        )
        assert not torch.allclose(out_a, out_b)

    def test_init__raises_when_embed_dim_not_divisible(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            CAPIPredictorTIMM(embed_dim=25, grid_size=4, num_heads=4)
