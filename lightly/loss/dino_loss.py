from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from lightly.models.modules.center import Center


class DINOLoss(Module):
    """
    Implementation of the loss described in 'Emerging Properties in
    Self-Supervised Vision Transformers'. [0]

    This implementation follows the code published by the authors. [1]
    It supports global and local image crops. A linear warmup schedule for the
    teacher temperature is implemented to stabilize training at the beginning.
    Centering is applied to the teacher output to avoid model collapse.

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: https://github.com/facebookresearch/dino

    Attributes:
        output_dim:
            Dimension of the model output.
        warmup_teacher_temp:
            Initial value of the teacher temperature. Should be decreased if the
            training loss does not decrease.
        teacher_temp:
            Final value of the teacher temperature after linear warmup. Values
            above 0.07 result in unstable behavior in most cases. Can be
            slightly increased to improve performance during finetuning.
        warmup_teacher_temp_epochs:
            Number of epochs for the teacher temperature warmup.
        student_temp:
            Temperature of the student.
        center_momentum:
            Momentum term for the center calculation.

    Examples:

        >>> # initialize loss function
        >>> loss_fn = DINOLoss(128)
        >>>
        >>> # generate a view of the images with a random transform
        >>> view = transform(images)
        >>>
        >>> # embed the view with a student and teacher model
        >>> teacher_out = teacher(view)
        >>> student_out = student(view)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn([teacher_out], [student_out], epoch=0)

    """

    def __init__(
        self,
        output_dim: int = 65536,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        center_mode: str = "mean",
    ):
        super().__init__()
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

        self._center = Center(
            size=(1, 1, output_dim),
            mode=center_mode,
            momentum=center_momentum,
            _register_buffer=False,
        )
        self.register_buffer("center", self._center.center)

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = torch.linspace(
            start=warmup_teacher_temp,
            end=teacher_temp,
            steps=warmup_teacher_temp_epochs,
        )

    # Center momentum is registered as property for backwards compatibility as it used
    # to be stored as attribute.
    @property
    def center_momentum(self) -> float:
        return self._center.momentum

    @center_momentum.setter
    def center_momentum(self, value: float) -> None:
        self._center.momentum = value

    def forward(
        self,
        teacher_out: List[Tensor],
        student_out: List[Tensor],
        epoch: int,
    ) -> Tensor:
        """Cross-entropy between softmax outputs of the teacher and student
        networks.

        Args:
            teacher_out:
                List of tensors with shape (batch_size, output_dim) containing features
                from the teacher model. Each tensor must represent one view of the
                batch.
            student_out:
                List of tensors with shape (batch_size, output_dim) containing features
                from the student model. Each tensor must represent one view of the
                batch.
            epoch:
                The current training epoch.
            update_center:
                If True, the center used for the teacher output is updated after the
                loss calculation.

        Returns:
            The average cross-entropy loss.

        """
        # get teacher temperature
        if epoch < self.warmup_teacher_temp_epochs:
            teacher_temp = self.teacher_temp_schedule[epoch]
        else:
            teacher_temp = self.teacher_temp

        teacher_out = torch.stack(teacher_out)
        t_out = F.softmax((teacher_out - self._center.value) / teacher_temp, dim=-1)

        student_out = torch.stack(student_out)
        s_out = F.log_softmax(student_out / self.student_temp, dim=-1)

        # calculate feature similarities where:
        # b = batch_size, t = n_views_teacher, s = n_views_student, d = output_dim
        # the diagonal is ignored as it contains features from the same views
        loss = -torch.einsum("tbd,sbd->ts", t_out, s_out)
        loss.fill_diagonal_(0)

        # number of loss terms, ignoring the diagonal
        n_terms = loss.numel() - loss.diagonal().numel()
        batch_size = teacher_out.shape[1]
        loss = loss.sum() / (n_terms * batch_size)

        self.update_center(teacher_out)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_out: Tensor) -> None:
        """Moving average update of the center used for the teacher output.

        Args:
            teacher_out:
                Tensor with shape (num_views, batch_size, output_dim) containing
                features from the teacher model.

        """
        self._center.update(teacher_out)
