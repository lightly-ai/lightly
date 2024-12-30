import copy
import typing
from typing import List

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lightly.loss import DINOLoss
from lightly.models.modules.center import Center
from lightly.models.utils import deactivate_requires_grad


class OriginalDINOLoss(nn.Module):
    """Copy paste from the original DINO paper. We use this to verify our
    implementation.

    The only change from the original code is that distributed training is no
    longer assumed.

    Source: https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/main_dino.py#L363

    """

    @typing.no_type_check
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    @typing.no_type_check
    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                s_out = F.log_softmax(student_out[v], dim=-1)
                loss = torch.sum(-q * s_out, dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @typing.no_type_check
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class TestDINOLoss:
    @pytest.mark.parametrize("batch_size", range(1, 4))
    @pytest.mark.parametrize("n_local", range(0, 4))
    @pytest.mark.parametrize("output_dim", range(1, 4))
    def test_different_input_sizes(
        self, batch_size: int, n_local: int, output_dim: int
    ) -> None:
        _run_test(batch_size=batch_size, n_local=n_local, output_dim=output_dim)

    @pytest.mark.parametrize("warmup_teacher_temp", [0.01, 0.04, 0.07])
    @pytest.mark.parametrize("teacher_temp", [0.01, 0.04, 0.07])
    @pytest.mark.parametrize("warmup_teacher_temp_epochs", [0, 1, 10])
    @pytest.mark.parametrize("epoch", [0, 1, 10, 20])
    def test_teacher_temperature_warmup(
        self,
        warmup_teacher_temp: float,
        teacher_temp: float,
        warmup_teacher_temp_epochs: int,
        epoch: int,
    ) -> None:
        _run_test(
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            epoch=epoch,
        )

    @pytest.mark.parametrize("student_temp", [0.05, 0.1, 0.2])
    @pytest.mark.parametrize("center_momentum", [0.5, 0.9, 0.95])
    def test_other_parameters(
        self,
        student_temp: float,
        center_momentum: float,
    ) -> None:
        _run_test(
            student_temp=student_temp,
            center_momentum=center_momentum,
        )


def test_center__equivalence() -> None:
    """Check that DINOLoss.update_center is equivalent to Center.update.

    TODO(Guarin, 08/24): Remove this test once DINOLoss uses Center internally.
    """
    criterion = DINOLoss(output_dim=32, center_momentum=0.9)
    center = Center(size=(1, 1, 32), momentum=0.9)
    x = torch.rand(2, 32)
    criterion.update_center(teacher_out=x)
    center.update(x=x)
    assert torch.allclose(criterion.center, center.value)


def _generate_output(
    batch_size: int = 2, n_views: int = 3, output_dim: int = 4, seed: int = 0
) -> List[Tensor]:
    """Returns a list of view representations.

    Example output:
        [
            torch.Tensor([img0_view0, img1_view0]),
            torch.Tensor([img0_view1, img1_view1])
        ]

    """
    torch.manual_seed(seed)
    out = []
    for _ in range(n_views):
        views = [torch.rand(output_dim) for _ in range(batch_size)]
        out.append(torch.stack(views))
    return out


def _run_test(
    batch_size: int = 3,
    n_global: int = 2,  # number of global views
    n_local: int = 6,  # number of local views
    output_dim: int = 4,
    warmup_teacher_temp: float = 0.04,
    teacher_temp: float = 0.04,
    warmup_teacher_temp_epochs: int = 30,
    student_temp: float = 0.1,
    center_momentum: float = 0.9,
    epoch: int = 0,
    n_epochs: int = 100,
) -> None:
    """Runs test with the given input parameters."""
    loss_fn = DINOLoss(
        output_dim=output_dim,
        warmup_teacher_temp=warmup_teacher_temp,
        teacher_temp=teacher_temp,
        warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
        student_temp=student_temp,
        center_momentum=center_momentum,
    )

    orig_loss_fn = OriginalDINOLoss(
        out_dim=output_dim,
        ncrops=n_global + n_local,
        teacher_temp=teacher_temp,
        warmup_teacher_temp=warmup_teacher_temp,
        warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
        nepochs=n_epochs,
        student_temp=student_temp,
        center_momentum=center_momentum,
    )

    # Create dummy single layer network. We use this to verify
    # that the gradient backprop works properly.
    teacher = torch.nn.Linear(output_dim, output_dim)
    deactivate_requires_grad(teacher)
    student = torch.nn.Linear(output_dim, output_dim)
    orig_teacher = copy.deepcopy(teacher)
    orig_student = copy.deepcopy(student)

    optimizer = torch.optim.SGD(student.parameters(), lr=1)
    orig_optimizer = torch.optim.SGD(orig_student.parameters(), lr=1)

    # Create fake output
    teacher_out = _generate_output(
        batch_size=batch_size,
        n_views=n_global,
        output_dim=output_dim,
        seed=0,
    )
    student_out = _generate_output(
        batch_size=batch_size,
        n_views=n_global + n_local,
        output_dim=output_dim,
        seed=1,
    )

    # Clone input tensors
    orig_teacher_out = torch.cat(teacher_out)
    orig_teacher_out = orig_teacher_out.detach().clone()
    orig_student_out = torch.cat(student_out)
    orig_student_out = orig_student_out.detach().clone()

    # Forward pass
    teacher_out = [teacher(view) for view in teacher_out]
    student_out = [student(view) for view in student_out]
    orig_teacher_out = orig_teacher(orig_teacher_out)
    orig_student_out = orig_student(orig_student_out)

    # Calculate loss
    loss = loss_fn(
        teacher_out=teacher_out,
        student_out=student_out,
        epoch=epoch,
    )
    orig_loss = orig_loss_fn(
        student_output=orig_student_out,
        teacher_output=orig_teacher_out,
        epoch=epoch,
    )

    # Backward pass and optimizer step
    optimizer.zero_grad()
    orig_optimizer.zero_grad()
    loss.backward()
    orig_loss.backward()
    optimizer.step()
    orig_optimizer.step()

    # Loss and loss center should be equal
    center = loss_fn.center.squeeze()
    orig_center = orig_loss_fn.center.squeeze()
    assert torch.allclose(center, orig_center)
    assert torch.allclose(loss, orig_loss)

    # Parameters of network should be equal after backward pass
    for param, orig_param in zip(student.parameters(), orig_student.parameters()):
        torch.testing.assert_close(param, orig_param)
    for param, orig_param in zip(teacher.parameters(), orig_teacher.parameters()):
        torch.testing.assert_close(param, orig_param)
