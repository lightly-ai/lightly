from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from lightly.models.modules import center
from lightly.models.modules.center import CENTER_MODE_TO_SUM_FUNCTION, MAX_ACCUMULATED


class DINOLoss(Module):
    """Implementation of the loss described in 'Emerging Properties in
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
        teacher_temp:
            Temperature parameter for the teacher network.
        student_temp:
            Temperature parameter for the student network.
        center:
            Center used for the teacher output. It is updated with a moving average
            during training.
        center_momentum:
            Momentum term for the center calculation.
        warmup_teacher_temp_epochs:
                Number of epochs for the warmup phase of the teacher temperature (for backward compatibility).
        teacher_temp_schedule:
            A linear schedule for the teacher temperature during the warmup phase (for backward compatibility).

    Examples:
        >>> # initialize loss function
        >>> loss_fn = DINOLoss(128)
        >>>
        >>> # generate two views of the images with a random transform
        >>> view0, view1 = transform(images), transform(images)
        >>>
        >>> # embed the views with a student and teacher model
        >>> teacher_out = [teacher(view0), teacher(view1)]
        >>> student_out = [student(view0), student(view1)]
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(teacher_out, student_out)
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
    ) -> None:
        """Initializes the DINOLoss Module.

        Args:
            center_mode:
                Mode for center calculation. Only 'mean' is supported.
            warmup_teacher_temp:
                Initial temperature for the teacher network (for backward compatibility).
            warmup_teacher_temp_epochs:
                Number of epochs for the warmup phase of the teacher temperature (for backward compatibility).
        """
        super().__init__()

        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

        # TODO(Guarin, 08/24): Refactor this to use the Center module directly once
        # we do a breaking change.
        if center_mode not in CENTER_MODE_TO_SUM_FUNCTION:
            raise ValueError(
                f"Unknown mode '{center_mode}'. Valid modes are "
                f"{sorted(CENTER_MODE_TO_SUM_FUNCTION.keys())}."
            )
        self._sum_fn = CENTER_MODE_TO_SUM_FUNCTION[center_mode]
        self.center: Tensor  # For mypy
        self.register_buffer("center", torch.zeros(1, 1, output_dim))
        self.center_momentum = center_momentum

        # Plain attributes instead of buffers: DDP broadcasts buffers from rank
        # zero before every forward, which would overwrite these rank-local sums.
        self._batch_sum: Tensor | None = None
        self._batch_num_elements = 0
        self._num_accumulated = 0
        self._warned_missing_update = False

        # comput the warmup teacher temperature internally for backward compatibility
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp_schedule = torch.linspace(
            start=warmup_teacher_temp,
            end=teacher_temp,
            steps=warmup_teacher_temp_epochs,
        )

    def forward(
        self,
        teacher_out: list[Tensor],
        student_out: list[Tensor],
        teacher_temp: float | None = None,
        epoch: int | None = None,
        *,
        update_center: bool = True,
    ) -> Tensor:
        """Cross-entropy between softmax outputs of the teacher and student networks.

        Args:
            teacher_out:
                List of tensors with shape (batch_size, output_dim) containing features
                from the teacher model. Each tensor must represent one view of the
                batch.
            student_out:
                List of tensors with shape (batch_size, output_dim) containing features
                from the student model. Each tensor must represent one view of the
                batch.
            teacher_temp:
                The temperature used for the teacher output. If None, the default
                temperature defined in __init__ is used.
            epoch:
                The current epoch for backward compatibility.
            update_center:
                Experimental: If True, the momentum update of the center is applied
                on every call. Set to False when training with gradient accumulation
                and call update_center once per optimizer step instead. The teacher
                output is accumulated regardless of this flag.

        Returns:
            The average cross-entropy loss.
        """
        # Get teacher temperature
        if teacher_temp is not None:
            teacher_temperature = torch.tensor(teacher_temp)
        elif epoch is not None:  # for backward compatibility
            if epoch < self.warmup_teacher_temp_epochs:
                teacher_temperature = self.teacher_temp_schedule[epoch]
            else:
                teacher_temperature = torch.tensor(self.teacher_temp)
        else:
            teacher_temperature = torch.tensor(self.teacher_temp)

        # Calculate cross-entropy loss.
        teacher_out_stacked = torch.stack(teacher_out)
        t_out: Tensor = F.softmax(
            (teacher_out_stacked - self.center) / teacher_temperature, dim=-1
        )
        student_out_stacked = torch.stack(student_out)
        s_out = F.log_softmax(student_out_stacked / self.student_temp, dim=-1)

        # Calculate feature similarities, ignoring the diagonal
        # b = batch_size, t = n_views_teacher, s = n_views_student, d = output_dim
        loss = -torch.einsum("tbd,sbd->ts", t_out, s_out)
        loss.fill_diagonal_(0)

        # Number of loss terms, ignoring the diagonal
        n_terms = loss.numel() - loss.diagonal().numel()
        batch_size = teacher_out_stacked.shape[1]

        if n_terms == 0:
            raise ValueError(
                "DINOLoss requires at least two views in total (the diagonal "
                "matching the same view index is excluded), but got "
                f"{len(teacher_out)} teacher view(s) and {len(student_out)} "
                "student view(s), which leaves no cross-view terms to compute "
                "the loss from."
            )

        loss = loss.sum() / (n_terms * batch_size)

        if self.training:
            self._accumulate_center(teacher_out_stacked)
            if update_center:
                self.update_center()

        return loss

    @torch.no_grad()
    def update_center(self, teacher_out: list[Tensor] | Tensor | None = None) -> None:
        """Moving average update of the center used for the teacher output.

        Runs a distributed collective, so all ranks must call this in lockstep.

        Args:
            teacher_out:
                Tensor with shape (num_views, batch_size, output_dim) containing
                features from the teacher model, or the list of per-view tensors
                passed to forward. If None, the center is updated from the features
                accumulated by previous forward passes only. The latter is the form
                to use when training with gradient accumulation.
        """
        if teacher_out is not None:
            if not isinstance(teacher_out, Tensor):
                teacher_out = torch.stack(teacher_out)
            self._accumulate_center(teacher_out)

        if self._batch_sum is None:
            return

        batch_center = center.reduce_mean(self._batch_sum / self._batch_num_elements)
        self.center.data = center.center_momentum(
            center=self.center, batch_center=batch_center, momentum=self.center_momentum
        )
        self._batch_sum = None
        self._batch_num_elements = 0
        self._num_accumulated = 0

    @torch.no_grad()
    def _accumulate_center(self, teacher_out: Tensor) -> None:
        """Accumulates the batch center without updating the center.

        Args:
            teacher_out:
                Tensor with shape (num_views, batch_size, output_dim) containing
                features from the teacher model.
        """
        batch_sum = self._sum_fn(x=teacher_out, dim=(0, 1))
        if self._batch_sum is None:
            self._batch_sum = batch_sum
        else:
            self._batch_sum += batch_sum
        self._batch_num_elements += teacher_out.shape[0] * teacher_out.shape[1]
        self._num_accumulated += 1

        if self._num_accumulated > MAX_ACCUMULATED and not self._warned_missing_update:
            self._warned_missing_update = True
            warnings.warn(
                f"{type(self).__name__} accumulated {self._num_accumulated} center "
                "updates without a call to update_center(). If you pass "
                "update_center=False for gradient accumulation, you must call "
                "update_center() once per optimizer step, otherwise the center "
                "stays frozen and the model may collapse.",
                UserWarning,
                stacklevel=2,
            )
