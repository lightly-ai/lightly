import pytest
import torch
from torch.nn import functional as F

from lightly.loss.ibot_loss import IBOTPatchLoss
from lightly.models.modules.center import sinkhorn_knopp


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

    def test__init__invalid_center_mode(self) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            IBOTPatchLoss(output_dim=2, center_mode="invalid")

    def test_sinkhorn_knopp(self) -> None:
        """Sinkhorn-Knopp centering replaces the mean centering of the teacher output.

        DINOv3 applies Sinkhorn-Knopp to the masked teacher tokens, see
        https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/ibot_patch_loss.py
        """
        teacher_temp, student_temp = 0.04, 0.2
        criterion = IBOTPatchLoss(
            output_dim=2,
            teacher_temp=teacher_temp,
            student_temp=student_temp,
            center_mode="sinkhorn_knopp",
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

        teacher_probs = sinkhorn_knopp(x=teacher_out, temperature=teacher_temp)
        cross_entropy = -(
            teacher_probs * F.log_softmax(student_out / student_temp, dim=-1)
        ).sum(dim=-1)
        weight = (1.0 / mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)).expand_as(
            mask
        )[mask]
        expected = (cross_entropy * weight).sum() / mask.shape[0]

        assert loss == pytest.approx(expected.item(), rel=1e-6)
        # No center is tracked with Sinkhorn-Knopp centering.
        assert torch.all(criterion.center.value == 0)
