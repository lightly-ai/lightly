import pytest
import torch

from lightly.loss.emp_ssl_loss import EMPSSLLoss


class TestEMPSSLLoss:
    @pytest.mark.parametrize("inv_coef", [0.0, 200.0])
    def test_forward_matches_reference(self, inv_coef: float) -> None:
        """Match the per-view objective and gradients from the EMP-SSL reference."""
        torch.manual_seed(0)
        z = torch.randn(3, 5, 4, dtype=torch.float64, requires_grad=True)
        eps = 0.2
        loss = EMPSSLLoss(tcr_eps=eps, inv_coef=inv_coef)(list(z.unbind()))

        # The reference minimizes negative coding rate, averaged over views.
        coding_rates = [
            torch.linalg.slogdet(
                torch.eye(4, dtype=z.dtype) + 4 / (5 * eps) * view.T @ view
            )[1]
            / 2
            for view in z
        ]
        similarity = torch.stack(
            [
                torch.nn.functional.cosine_similarity(view, z.mean(0)).mean()
                for view in z
            ]
        ).mean()
        expected = -torch.stack(coding_rates).mean() - inv_coef * similarity

        torch.testing.assert_close(loss, expected)
        (gradient,) = torch.autograd.grad(loss, z, retain_graph=True)
        (expected_gradient,) = torch.autograd.grad(expected, z)
        torch.testing.assert_close(gradient, expected_gradient)

    def test_prefers_diverse_embeddings(self) -> None:
        """With identical views, spreading unit embeddings must lower the loss."""
        loss_fn = EMPSSLLoss()
        diverse = torch.eye(4)
        collapsed = diverse[:1].expand(4, -1)

        assert loss_fn([diverse, diverse]) < loss_fn([collapsed, collapsed])

    def test_forward(self) -> None:
        bs = 512
        dim = 128
        num_views = 100

        loss_fn = EMPSSLLoss()
        x = [torch.randn(bs, dim)] * num_views

        loss_fn(x)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_forward_cuda(self) -> None:
        bs = 512
        dim = 128
        num_views = 100

        loss_fn = EMPSSLLoss().cuda()
        x = x = [torch.randn(bs, dim).cuda()] * num_views

        loss_fn(x)
