import pytest
import torch
import torch.nn.functional as F

from lightly.loss.ibot_loss import IBOTPlusPlusPatchLoss


class TestIBOTPlusPlusPatchLoss:
    def test_forward_rank3(self) -> None:
        torch.manual_seed(0)
        criterion = IBOTPlusPlusPatchLoss(output_dim=8)
        teacher_out = torch.randn(4, 196, 8)
        student_out = torch.randn(4, 196, 8)
        loss = criterion(teacher_out=teacher_out, student_out=student_out)
        assert loss.ndim == 0
        assert loss.isfinite()

    def test_backward_rank3(self) -> None:
        torch.manual_seed(0)
        criterion = IBOTPlusPlusPatchLoss(output_dim=8)
        teacher_out = torch.randn(2, 10, 8)
        student_out = torch.randn(2, 10, 8, requires_grad=True)
        loss = criterion(teacher_out=teacher_out, student_out=student_out)
        loss.backward()
        assert student_out.grad is not None
        assert student_out.grad.isfinite().all()

    def test_matches_manual_cross_entropy(self) -> None:
        # With a zero center the loss should match a plain mean cross-entropy.
        torch.manual_seed(42)
        B, N, K = 2, 4, 6
        teacher_temp = 0.04
        student_temp = 0.1
        criterion = IBOTPlusPlusPatchLoss(
            output_dim=K,
            teacher_temp=teacher_temp,
            student_temp=student_temp,
            center_momentum=0.9,
        )
        # Zero out the center so it has no effect.
        criterion.center.center.zero_()

        teacher_out = torch.randn(B, N, K)
        student_out = torch.randn(B, N, K)

        loss = criterion(teacher_out=teacher_out, student_out=student_out)

        # Manual reference.
        t_flat = teacher_out.flatten(0, 1)
        s_flat = student_out.flatten(0, 1)
        q = F.softmax(t_flat / teacher_temp, dim=-1)
        log_p = F.log_softmax(s_flat / student_temp, dim=-1)
        expected = (-torch.sum(q * log_p, dim=-1)).view(B, N).mean(dim=1).mean()

        assert loss == pytest.approx(expected.item(), rel=1e-5)

    def test_center_updates_from_all_tokens(self) -> None:
        torch.manual_seed(0)
        K = 4
        criterion = IBOTPlusPlusPatchLoss(
            output_dim=K,
            center_momentum=0.9,
        )
        B, N = 3, 5
        teacher_out = torch.randn(B, N, K)
        student_out = torch.randn(B, N, K)

        criterion(teacher_out=teacher_out, student_out=student_out)

        # Center initialized to zero, so after one update:
        # center = 0 * 0.9 + mean(all_tokens) * 0.1
        expected = 0.1 * teacher_out.flatten(0, 1).mean(0, keepdim=True)
        assert torch.allclose(criterion.center.value, expected, atol=1e-6)

    def test_rank2_with_mask(self) -> None:
        torch.manual_seed(1)
        B, N, K = 3, 7, 8
        teacher_out = torch.randn(B, N, K)
        student_out = torch.randn(B, N, K)

        criterion_3d = IBOTPlusPlusPatchLoss(output_dim=K)
        loss_3d = criterion_3d(teacher_out=teacher_out, student_out=student_out)

        criterion_2d = IBOTPlusPlusPatchLoss(output_dim=K)
        mask = torch.zeros(B, N, dtype=torch.bool)
        loss_2d = criterion_2d(
            teacher_out=teacher_out.flatten(0, 1),
            student_out=student_out.flatten(0, 1),
            mask=mask,
        )

        assert loss_2d == pytest.approx(loss_3d.item(), rel=1e-5)

    def test_visible_loss_weight_zero_matches_masked_only(self) -> None:
        # With visible_loss_weight=0 and a mask, iBOT++ must reduce to the
        # original iBOT masked-only loss (per-image mean over masked tokens).
        torch.manual_seed(7)
        B, N, K = 3, 8, 6
        teacher_temp, student_temp = 0.04, 0.1
        criterion = IBOTPlusPlusPatchLoss(
            output_dim=K, teacher_temp=teacher_temp, student_temp=student_temp
        )
        # Zero out the center so it has no effect.
        criterion.center.center.zero_()

        teacher_out = torch.randn(B, N, K)
        student_out = torch.randn(B, N, K)
        mask = torch.rand(B, N) < 0.4
        mask[:, 0] = True  # ensure at least one masked token per image

        loss = criterion(
            teacher_out=teacher_out,
            student_out=student_out,
            mask=mask,
            visible_loss_weight=0.0,
        )

        # Manual masked-only reference.
        q = F.softmax(teacher_out / teacher_temp, dim=-1)
        log_p = F.log_softmax(student_out / student_temp, dim=-1)
        ce = -(q * log_p).sum(dim=-1)  # (B, N)
        m = mask.float()
        expected = ((ce * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)).mean()

        assert loss == pytest.approx(expected.item(), rel=1e-5)

    def test_visible_loss_weight_decouples_masked_and_visible(self) -> None:
        # With visible_loss_weight=0 the visible tokens must not contribute to
        # the gradient, while masked tokens do.
        torch.manual_seed(3)
        B, N, K = 2, 6, 5
        teacher_out = torch.randn(B, N, K)
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[:, :3] = True  # first half masked, second half visible

        criterion = IBOTPlusPlusPatchLoss(output_dim=K)
        criterion.center.center.zero_()

        student_out = torch.randn(B, N, K, requires_grad=True)
        loss = criterion(
            teacher_out=teacher_out,
            student_out=student_out,
            mask=mask,
            visible_loss_weight=0.0,
        )
        loss.backward()

        assert student_out.grad is not None
        # Visible tokens get no gradient when their weight is zero.
        assert torch.allclose(
            student_out.grad[:, 3:], torch.zeros_like(student_out.grad[:, 3:]), atol=1e-7
        )
        # Masked tokens do receive gradient.
        assert student_out.grad[:, :3].abs().sum() > 0

    def test_invalid_shapes_raise(self) -> None:
        criterion = IBOTPlusPlusPatchLoss(output_dim=4)

        # Mismatched teacher/student shapes.
        with pytest.raises(ValueError, match="same shape"):
            criterion(
                teacher_out=torch.randn(2, 5, 4),
                student_out=torch.randn(2, 6, 4),
            )

        # Rank 4 input.
        with pytest.raises(ValueError, match="rank 2 or 3"):
            criterion(
                teacher_out=torch.randn(2, 5, 3, 4),
                student_out=torch.randn(2, 5, 3, 4),
            )

        # Rank 2 without mask.
        with pytest.raises(ValueError, match="mask is required"):
            criterion(
                teacher_out=torch.randn(10, 4),
                student_out=torch.randn(10, 4),
            )

        # Rank 2 with incompatible mask (B*N not divisible by B).
        with pytest.raises(ValueError, match="not divisible"):
            criterion(
                teacher_out=torch.randn(7, 4),
                student_out=torch.randn(7, 4),
                mask=torch.zeros(3, 2, dtype=torch.bool),
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_cuda_forward(self) -> None:
        torch.manual_seed(0)
        criterion = IBOTPlusPlusPatchLoss(output_dim=8).cuda()
        teacher_out = torch.randn(4, 16, 8).cuda()
        student_out = torch.randn(4, 16, 8).cuda()
        loss = criterion(teacher_out=teacher_out, student_out=student_out)
        assert loss.isfinite()
        assert loss.device.type == "cuda"
