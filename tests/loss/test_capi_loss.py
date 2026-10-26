from __future__ import annotations

import pytest
import torch

from lightly.loss import CAPILoss
from lightly.loss.capi_loss import sinkhorn_knopp


class TestCAPILoss:
    def test_forward__returns_positive_scalar(self) -> None:
        criterion = CAPILoss(sinkhorn_iterations=3)
        teacher_logits = torch.randn(4, 5, 32)
        student_logits = torch.randn(4, 5, 32)
        loss = criterion(teacher_logits=teacher_logits, student_logits=student_logits)
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_forward__gradient_flows_to_student_not_teacher(self) -> None:
        criterion = CAPILoss()
        teacher_logits = torch.randn(2, 3, 8, requires_grad=True)
        student_logits = torch.randn(2, 3, 8, requires_grad=True)
        loss = criterion(teacher_logits=teacher_logits, student_logits=student_logits)
        loss.backward()
        assert student_logits.grad is not None
        # Teacher targets are produced under no_grad, so no gradient reaches them.
        assert teacher_logits.grad is None

    def test_forward__matched_prediction_lower_than_mismatched(self) -> None:
        criterion = CAPILoss()
        teacher_logits = torch.randn(8, 4, 16)
        matched = criterion(
            teacher_logits=teacher_logits, student_logits=teacher_logits.clone()
        )
        mismatched = criterion(
            teacher_logits=teacher_logits, student_logits=torch.randn(8, 4, 16)
        )
        assert matched < mismatched

    def test_forward__teacher_index_returns_positive_scalar(self) -> None:
        criterion = CAPILoss()
        teacher_logits = torch.randn(2, 6, 16)
        student_logits = torch.randn(2, 3, 16)
        teacher_index = torch.tensor([[0, 2, 4], [1, 3, 5]])
        loss = criterion(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            teacher_index=teacher_index,
        )
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_forward__teacher_index_normalizes_over_all_tokens(self) -> None:
        # The teacher_index path equals normalizing over all teacher tokens and
        # then gathering the assignments at the predicted positions.
        criterion = CAPILoss()
        teacher_logits = torch.randn(2, 6, 16)
        student_logits = torch.randn(2, 3, 16)
        teacher_index = torch.tensor([[0, 2, 4], [1, 3, 5]])
        loss = criterion(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            teacher_index=teacher_index,
        )
        assignments = sinkhorn_knopp(teacher_logits / 0.06, iterations=3)
        index = teacher_index.unsqueeze(-1).expand(-1, -1, assignments.shape[-1])
        selected = torch.gather(assignments, 1, index)
        log_probs = torch.log_softmax(student_logits / 0.12, dim=-1)
        expected = -torch.sum(selected * log_probs, dim=-1).mean()
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_forward__raises_when_distributed_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.distributed, "is_available", lambda: False)
        with pytest.raises(ValueError, match="torch.distributed is not available"):
            CAPILoss(gather_distributed=True)


class TestSinkhornKnopp:
    def test__returns_distribution_over_clusters(self) -> None:
        logits = torch.randn(6, 5, 10)  # (batch_size, sequence_length, num_clusters)
        assignments = sinkhorn_knopp(logits, iterations=3)
        assert assignments.shape == (6, 5, 10)
        # Each token's assignment is a distribution over the clusters.
        sums = assignments.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test__is_positionwise(self) -> None:
        # Positions are normalized independently, so selecting a position before or
        # after Sinkhorn gives the same result.
        logits = torch.randn(4, 3, 7)
        full = sinkhorn_knopp(logits, iterations=3)
        single = sinkhorn_knopp(logits[:, 1:2, :], iterations=3)
        assert torch.allclose(full[:, 1:2, :], single, atol=1e-5)
