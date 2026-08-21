from __future__ import annotations

import pytest
import torch

from lightly.models.modules.world_model import predictor as predictor_module
from lightly.models.modules.world_model.predictor import LatentDynamicsPredictor


def _predictor(**kwargs: object) -> LatentDynamicsPredictor:
    defaults: dict[str, object] = dict(
        num_frames=4,
        input_dim=16,
        hidden_dim=32,
        depth=2,
        num_heads=4,
    )
    defaults.update(kwargs)
    return LatentDynamicsPredictor(**defaults)  # type: ignore[arg-type]


class TestLatentDynamicsPredictor:
    def test_forward(self) -> None:
        predictor = _predictor()
        out = predictor(torch.randn(3, 4, 16), action_emb=torch.randn(3, 4, 16))
        assert out.shape == (3, 4, 16)

    def test_forward__output_dim(self) -> None:
        predictor = _predictor(output_dim=8)
        out = predictor(torch.randn(3, 4, 16), action_emb=torch.randn(3, 4, 16))
        assert out.shape == (3, 4, 8)

    def test_forward__shorter_than_context_window(self) -> None:
        predictor = _predictor()
        out = predictor(torch.randn(3, 2, 16), action_emb=torch.randn(3, 2, 16))
        assert out.shape == (3, 2, 16)

    def test_forward__is_causal(self) -> None:
        """A change at frame t must not change the outputs before t."""
        torch.manual_seed(0)
        predictor = _predictor().eval()
        emb = torch.randn(2, 4, 16)
        action_emb = torch.randn(2, 4, 16)

        with torch.no_grad():
            out = predictor(emb, action_emb=action_emb)
            perturbed = emb.clone()
            perturbed[:, 2] += 10.0
            out_perturbed = predictor(perturbed, action_emb=action_emb)

        assert torch.allclose(out[:, :2], out_perturbed[:, :2], atol=1e-5)
        assert not torch.allclose(out[:, 2], out_perturbed[:, 2], atol=1e-5)

    def test_forward__action_is_causal(self) -> None:
        """The action at frame t must not change the outputs before t."""
        torch.manual_seed(0)
        predictor = _trained_predictor()
        emb = torch.randn(2, 4, 16)
        action_emb = torch.randn(2, 4, 16)

        with torch.no_grad():
            out = predictor(emb, action_emb=action_emb)
            perturbed = action_emb.clone()
            perturbed[:, 2] += 10.0
            out_perturbed = predictor(emb, action_emb=perturbed)

        assert torch.allclose(out[:, :2], out_perturbed[:, :2], atol=1e-5)
        assert not torch.allclose(out[:, 2], out_perturbed[:, 2], atol=1e-5)

    def test_forward__action_changes_prediction(self) -> None:
        predictor = _trained_predictor()
        with torch.no_grad():
            emb = torch.randn(2, 4, 16)
            out_a = predictor(emb, action_emb=torch.zeros(2, 4, 16))
            out_b = predictor(emb, action_emb=torch.ones(2, 4, 16))
        assert not torch.allclose(out_a, out_b, atol=1e-5)

    def test_forward__adaln_zero_starts_as_identity_conditioning(self) -> None:
        """At initialization the gates are zero, so the action cannot matter."""
        torch.manual_seed(0)
        predictor = _predictor().eval()
        emb = torch.randn(2, 4, 16)
        with torch.no_grad():
            out_a = predictor(emb, action_emb=torch.zeros(2, 4, 16))
            out_b = predictor(emb, action_emb=torch.ones(2, 4, 16))
        assert torch.allclose(out_a, out_b, atol=1e-6)

    def test_forward__too_many_frames(self) -> None:
        predictor = _predictor()
        with pytest.raises(ValueError, match="context window"):
            predictor(torch.randn(3, 8, 16), action_emb=torch.randn(3, 8, 16))

    def test_forward__invalid_embeddings_shape(self) -> None:
        predictor = _predictor()
        with pytest.raises(ValueError, match="embeddings must have shape"):
            predictor(torch.randn(3, 4, 2, 16), action_emb=torch.randn(3, 4, 16))

    def test_forward__mismatched_action_shape(self) -> None:
        predictor = _predictor()
        with pytest.raises(ValueError, match="action_emb must have shape"):
            predictor(torch.randn(3, 4, 16), action_emb=torch.randn(3, 2, 16))

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(num_frames=0), "num_frames must be"),
            (dict(hidden_dim=30, num_heads=4), "divisible by num_heads"),
        ],
    )
    def test_init__invalid(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _predictor(**kwargs)

    def test_backward_pass(self) -> None:
        predictor = _predictor()
        emb = torch.randn(3, 4, 16, requires_grad=True)
        predictor(emb, action_emb=torch.randn(3, 4, 16)).sum().backward()
        assert emb.grad is not None
        assert emb.grad.shape == emb.shape


class TestRollout:
    def test_rollout(self) -> None:
        predictor = _predictor()
        rolled = predictor.rollout(
            torch.randn(3, 2, 16), action_emb=torch.randn(3, 6, 16), steps=4
        )
        assert rolled.shape == (3, 4, 16)

    def test_rollout__single_step_matches_forward(self) -> None:
        """One rollout step is the last entry of a forward pass."""
        torch.manual_seed(0)
        predictor = _predictor().eval()
        emb = torch.randn(2, 3, 16)
        action_emb = torch.randn(2, 3, 16)
        with torch.no_grad():
            rolled = predictor.rollout(emb, action_emb=action_emb, steps=1)
            forwarded = predictor(emb, action_emb=action_emb)
        assert torch.allclose(rolled[:, 0], forwarded[:, -1], atol=1e-6)

    def test_rollout__slides_context_window(self) -> None:
        """Rolling past the context window must not raise."""
        predictor = _predictor(num_frames=3)
        rolled = predictor.rollout(
            torch.randn(2, 3, 16), action_emb=torch.randn(2, 10, 16), steps=6
        )
        assert rolled.shape == (2, 6, 16)

    def test_rollout__too_few_actions(self) -> None:
        predictor = _predictor()
        with pytest.raises(ValueError, match="action_emb must cover"):
            predictor.rollout(
                torch.randn(3, 2, 16), action_emb=torch.randn(3, 2, 16), steps=4
            )

    def test_rollout__invalid_steps(self) -> None:
        predictor = _predictor()
        with pytest.raises(ValueError, match="steps must be"):
            predictor.rollout(
                torch.randn(3, 2, 16), action_emb=torch.randn(3, 6, 16), steps=0
            )

    def test_rollout__gradients_reach_the_context(self) -> None:
        predictor = _predictor()
        emb = torch.randn(3, 2, 16, requires_grad=True)
        predictor.rollout(
            emb, action_emb=torch.randn(3, 6, 16), steps=4
        ).sum().backward()
        assert emb.grad is not None
        assert not torch.allclose(emb.grad, torch.zeros_like(emb.grad))


class TestAttentionFallback:
    """The package declares torch>=1.10, and the fused kernel arrives in 2.0."""

    @pytest.mark.skipif(
        not predictor_module._FUSED_ATTENTION_AVAILABLE,
        reason="torch has no fused attention kernel before 2.0.",
    )
    def test_manual_attention_matches_the_fused_kernel(self) -> None:
        torch.manual_seed(0)
        query, key, value = (torch.randn(2, 4, 6, 8) for _ in range(3))
        mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))

        fused_attention = getattr(torch.nn.functional, "scaled_dot_product_attention")
        fused = fused_attention(query, key, value, attn_mask=mask, dropout_p=0.0)
        manual = predictor_module._manual_attention(query, key, value, mask, 0.0, False)
        assert torch.allclose(fused, manual, atol=1e-5)

    def test_predictor_matches_with_the_fallback_forced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch.manual_seed(0)
        predictor = _predictor().eval()
        emb, action_emb = torch.randn(2, 4, 16), torch.randn(2, 4, 16)
        with torch.no_grad():
            fused = predictor(emb, action_emb=action_emb)
            monkeypatch.setattr(predictor_module, "_FUSED_ATTENTION_AVAILABLE", False)
            manual = predictor(emb, action_emb=action_emb)
        assert torch.allclose(fused, manual, atol=1e-5)

    def test_manual_path_is_causal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The mask must mean the same thing on both paths."""
        monkeypatch.setattr(predictor_module, "_FUSED_ATTENTION_AVAILABLE", False)
        torch.manual_seed(0)
        predictor = _predictor().eval()
        emb = torch.randn(2, 4, 16)
        action_emb = torch.randn(2, 4, 16)

        with torch.no_grad():
            out = predictor(emb, action_emb=action_emb)
            perturbed = emb.clone()
            perturbed[:, 2] += 10.0
            out_perturbed = predictor(perturbed, action_emb=action_emb)

        assert torch.allclose(out[:, :2], out_perturbed[:, :2], atol=1e-5)
        assert not torch.allclose(out[:, 2], out_perturbed[:, 2], atol=1e-5)


def _trained_predictor() -> LatentDynamicsPredictor:
    """Train briefly so the AdaLN-Zero gates leave their zero initialization.

    Until they do, the conditioning path is inactive by construction and any
    test about the effect of an action would pass for the wrong reason.
    """
    torch.manual_seed(0)
    predictor = _predictor()
    emb = torch.randn(2, 4, 16)
    target = torch.randn(2, 4, 16)
    optimizer = torch.optim.SGD(predictor.parameters(), lr=0.1)
    for _ in range(5):
        loss = (
            (predictor(emb, action_emb=torch.randn(2, 4, 16)) - target).square().mean()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return predictor.eval()


@pytest.mark.slow
def test_predictor__overfits_linear_dynamics() -> None:
    """The predictor must actually learn action-conditioned dynamics.

    Every other test here pins a shape, a gradient or a masking property, and
    all of them would still pass if the predictor stopped learning. This one
    fits a known linear system ``z[t + 1] = A z[t] + B a[t]`` and checks that
    the error drops far below what predicting the current frame achieves.
    """
    torch.manual_seed(0)
    dim, num_frames, batch_size = 8, 4, 64
    transition = torch.randn(dim, dim) * 0.3
    control = torch.randn(dim, dim) * 0.3

    actions = torch.randn(batch_size, num_frames, dim)
    frames = [torch.randn(batch_size, dim)]
    for t in range(num_frames - 1):
        frames.append(frames[-1] @ transition.T + actions[:, t] @ control.T)
    emb = torch.stack(frames, dim=1)

    predictor = LatentDynamicsPredictor(
        num_frames=num_frames, input_dim=dim, hidden_dim=64, depth=2, num_heads=4
    )
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=3e-3)
    for _ in range(300):
        predicted = predictor(emb[:, :-1], action_emb=actions[:, :-1])
        loss = (predicted - emb[:, 1:]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # The trivial baseline copies the current frame to the next one.
    baseline = (emb[:, :-1] - emb[:, 1:]).square().mean()
    assert loss.item() < 0.1 * baseline.item()
