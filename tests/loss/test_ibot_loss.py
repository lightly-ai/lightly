import pytest
import torch

from lightly.loss.ibot_loss import IBOTPatchLoss


class TestIBOTPatchLoss:
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_forward(self, device: str) -> None:
        if not torch.cuda.is_available() and device == "cuda":
            pytest.skip("CUDA not available")

        criterion = IBOTPatchLoss(
            output_dim=2,
            teacher_temp=0.1,
            student_temp=0.2,
            center_mode="mean",
            center_momentum=0.9,
        )
        teacher_out = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        student_out = torch.tensor([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]])
        mask = torch.tensor(
            [
                [[True, False], [True, False]],
                [[False, False], [False, True]],
                [[False, False], [False, False]],
            ]
        )

        loss = criterion.forward(
            teacher_out=teacher_out, student_out=student_out, mask=mask
        )
        assert loss == pytest.approx(0.4057, rel=0.0001)
        expected_center = 0.1 * teacher_out.mean(0)
        assert torch.all(torch.isclose(criterion.center.value, expected_center))
        # Loss value was calculated with the original implementation from:
        # https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/ibot_patch_loss.py
        #
        # Code:
        # orig_criterion = iBOTPatchLoss(patch_out_dim=2, student_temp=0.2)
        # orig_t_center = orig_criterion.softmax_center_teacher(teacher_out, 0.1)
        # orig_loss = orig_criterion.forward_masked(
        #     student_patch_tokens_masked=student_out,
        #     teacher_patch_tokens_masked=orig_t_center,
        #     student_masks_flat=mask.flatten(start_dim=1),
        # )
