from typing import List

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

        criterion = criterion.to(device)
        teacher_out = teacher_out.to(device)
        student_out = student_out.to(device)
        mask = mask.to(device)

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


class TestIBOTPatchLossCenterUpdate:
    @pytest.mark.parametrize("num_micro_batches", [1, 2, 4])
    def test_gradient_accumulation__equivalent_to_single_batch(
        self, num_micro_batches: int
    ) -> None:
        """Deferring the update matches a single forward over the full batch."""
        torch.manual_seed(0)
        output_dim = 4
        criterion = IBOTPatchLoss(output_dim=output_dim, center_momentum=0.9)
        single = IBOTPatchLoss(output_dim=output_dim, center_momentum=0.9)

        micro_batches = []
        for _ in range(num_micro_batches):
            mask = torch.ones(2, 2, 2, dtype=torch.bool)
            teacher_out = torch.rand(int(mask.sum()), output_dim)
            student_out = torch.rand(int(mask.sum()), output_dim)
            micro_batches.append((teacher_out, student_out, mask))

        for teacher_out, student_out, mask in micro_batches:
            criterion(
                teacher_out=teacher_out,
                student_out=student_out,
                mask=mask,
                update_center=False,
            )
        criterion.center.update()

        single(
            teacher_out=torch.cat([m[0] for m in micro_batches], dim=0),
            student_out=torch.cat([m[1] for m in micro_batches], dim=0),
            mask=torch.cat([m[2] for m in micro_batches], dim=0),
        )

        assert torch.allclose(criterion.center.value, single.center.value)

    def test_eval__does_not_update_center(self) -> None:
        criterion = IBOTPatchLoss(output_dim=4)
        criterion.eval()
        mask = torch.ones(2, 2, 2, dtype=torch.bool)
        criterion(
            teacher_out=torch.rand(int(mask.sum()), 4),
            student_out=torch.rand(int(mask.sum()), 4),
            mask=mask,
        )
        assert torch.all(criterion.center.value == 0)

    def test_state_dict__accumulator_not_persisted(self) -> None:
        criterion = IBOTPatchLoss(output_dim=4)
        assert set(criterion.state_dict().keys()) == {"center.center"}

    @pytest.mark.parametrize("masked_per_image", [[2, 6], [1, 8, 3]])
    def test_gradient_accumulation__unequal_masked_tokens(
        self, masked_per_image: List[int]
    ) -> None:
        """iBOT masks a variable number of tokens, so sizes differ by construction."""
        torch.manual_seed(0)
        output_dim = 4
        criterion = IBOTPatchLoss(output_dim=output_dim, center_momentum=0.9)
        single = IBOTPatchLoss(output_dim=output_dim, center_momentum=0.9)

        micro_batches = []
        for num_masked in masked_per_image:
            mask = torch.zeros(2, 4, 4, dtype=torch.bool)
            mask.view(2, -1)[:, :num_masked] = True
            teacher_out = torch.rand(int(mask.sum()), output_dim)
            student_out = torch.rand(int(mask.sum()), output_dim)
            micro_batches.append((teacher_out, student_out, mask))

        for teacher_out, student_out, mask in micro_batches:
            criterion(
                teacher_out=teacher_out,
                student_out=student_out,
                mask=mask,
                update_center=False,
            )
        criterion.center.update()

        single(
            teacher_out=torch.cat([m[0] for m in micro_batches], dim=0),
            student_out=torch.cat([m[1] for m in micro_batches], dim=0),
            mask=torch.cat([m[2] for m in micro_batches], dim=0),
        )

        assert torch.allclose(criterion.center.value, single.center.value)
