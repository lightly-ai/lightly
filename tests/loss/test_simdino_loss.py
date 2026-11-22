import pytest
import torch
import torch.nn.functional as F

from lightly.loss.simdino_loss import SimDINOLoss


def _reference_loss(
    teacher: torch.Tensor, student: torch.Tensor, eps: float, compression_coeff: float
) -> torch.Tensor:
    """The SimDINO objective written out per view pair, as in the reference code."""
    num_teacher_views, batch_size, dim = teacher.shape
    similarities = [
        F.cosine_similarity(teacher[t], student[s], dim=-1).mean()
        for t in range(num_teacher_views)
        for s in range(len(student))
        if t != s
    ]
    compression = torch.stack(similarities).mean()
    coding_rates = [
        torch.linalg.slogdet(
            torch.eye(dim, dtype=teacher.dtype)
            + dim / (batch_size * eps) * view.T @ view
        )[1]
        / 2
        for view in (student[:num_teacher_views] + teacher) / 2
    ]
    gamma = (dim + batch_size) / (dim * batch_size)
    expansion = gamma * torch.stack(coding_rates).mean()
    return -compression_coeff * compression - expansion


class TestSimDINOLoss:
    @pytest.mark.parametrize("num_local_views", [0, 4])
    @pytest.mark.parametrize("eps, compression_coeff", [(0.5, 1.0), (0.2, 0.5)])
    def test_forward_matches_reference(
        self, num_local_views: int, eps: float, compression_coeff: float
    ) -> None:
        """Match the loss and gradients of the objective from the reference code."""
        torch.manual_seed(0)
        teacher = F.normalize(torch.randn(2, 8, 6, dtype=torch.float64), dim=-1)
        student = F.normalize(
            torch.randn(2 + num_local_views, 8, 6, dtype=torch.float64), dim=-1
        ).requires_grad_()
        loss_fn = SimDINOLoss(eps=eps, compression_coeff=compression_coeff)

        loss = loss_fn(list(teacher.unbind()), list(student.unbind()))
        expected = _reference_loss(teacher, student, eps, compression_coeff)

        torch.testing.assert_close(loss, expected)
        (gradient,) = torch.autograd.grad(loss, student, retain_graph=True)
        (expected_gradient,) = torch.autograd.grad(expected, student)
        torch.testing.assert_close(gradient, expected_gradient)

    def test_prefers_diverse_embeddings(self) -> None:
        """With matching views, spread out embeddings must have a lower loss."""
        loss_fn = SimDINOLoss()
        diverse = torch.eye(4)
        collapsed = diverse[:1].expand(4, -1)

        loss_diverse = loss_fn([diverse, diverse], [diverse, diverse])
        loss_collapsed = loss_fn([collapsed, collapsed], [collapsed, collapsed])

        assert loss_diverse < loss_collapsed

    def test_skips_matching_view(self) -> None:
        """A student view is not compared with the teacher output of the same view.

        Views a and b are orthogonal, so the only non-zero similarities are the
        matching (a, a) and (b, b) pairs. Skipping them leaves a compression term of
        zero, and the loss must not depend on its coefficient.
        """
        a = torch.eye(4)[:, :2].T.repeat(2, 1)  # rows e0, e1, e0, e1
        b = torch.eye(4)[:, 2:].T.repeat(2, 1)  # rows e2, e3, e2, e3
        views = [a, b]
        loss_with = SimDINOLoss(compression_coeff=1.0)(views, views)
        loss_without = SimDINOLoss(compression_coeff=0.0)(views, views)
        torch.testing.assert_close(loss_with, loss_without)

    def test_backward(self) -> None:
        torch.manual_seed(0)
        teacher = [F.normalize(torch.randn(4, 3), dim=-1) for _ in range(2)]
        student = [
            F.normalize(torch.randn(4, 3), dim=-1).requires_grad_() for _ in range(3)
        ]
        SimDINOLoss()(teacher, student).backward()
        assert all(view.grad is not None for view in student)

    def test_gather_distributed_single_process(self) -> None:
        """Without an initialized process group, gathering changes nothing."""
        torch.manual_seed(0)
        teacher = [F.normalize(torch.randn(4, 3), dim=-1) for _ in range(2)]
        student = [F.normalize(torch.randn(4, 3), dim=-1) for _ in range(3)]
        torch.testing.assert_close(
            SimDINOLoss(gather_distributed=True)(teacher, student),
            SimDINOLoss()(teacher, student),
        )

    def test_invalid_eps(self) -> None:
        with pytest.raises(ValueError, match="eps must be positive"):
            SimDINOLoss(eps=0.0)

    def test_more_teacher_than_student_views(self) -> None:
        views = [torch.randn(4, 3) for _ in range(2)]
        with pytest.raises(ValueError, match="at least the teacher's views"):
            SimDINOLoss()(views, views[:1])

    def test_single_view(self) -> None:
        view = [torch.randn(4, 3)]
        with pytest.raises(ValueError, match="at least two views"):
            SimDINOLoss()(view, view)
