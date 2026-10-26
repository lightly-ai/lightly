from __future__ import annotations

import pytest
import torch

from lightly.models.modules.world_model.conditioning import ActionEncoder


class TestActionEncoder:
    def test_forward(self) -> None:
        action_encoder = ActionEncoder(action_dim=2, output_dim=16)
        assert action_encoder(torch.randn(8, 4, 2)).shape == (8, 4, 16)

    def test_forward__is_applied_per_timestep(self) -> None:
        """The same action must map to the same embedding at any timestep."""
        torch.manual_seed(0)
        action_encoder = ActionEncoder(action_dim=2, output_dim=16).eval()
        actions = torch.randn(1, 1, 2).expand(3, 5, 2)
        with torch.no_grad():
            out = action_encoder(actions)
        assert torch.allclose(out, out[:1, :1].expand_as(out), atol=1e-6)

    def test_forward__invalid_shape(self) -> None:
        action_encoder = ActionEncoder(action_dim=2, output_dim=16)
        with pytest.raises(ValueError, match="actions must have shape"):
            action_encoder(torch.randn(8, 2))

    def test_init__hidden_dim_default(self) -> None:
        action_encoder = ActionEncoder(action_dim=2, output_dim=16)
        assert action_encoder.mlp[0].out_features == 64

    def test_init__hidden_dim(self) -> None:
        action_encoder = ActionEncoder(action_dim=2, output_dim=16, hidden_dim=8)
        assert action_encoder.mlp[0].out_features == 8

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(action_dim=0, output_dim=16), "action_dim must be"),
            (dict(action_dim=2, output_dim=0), "output_dim must be"),
        ],
    )
    def test_init__invalid_dims(self, kwargs: dict[str, int], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ActionEncoder(**kwargs)

    def test_backward_pass(self) -> None:
        action_encoder = ActionEncoder(action_dim=2, output_dim=16)
        actions = torch.randn(8, 4, 2, requires_grad=True)
        action_encoder(actions).sum().backward()
        assert actions.grad is not None
