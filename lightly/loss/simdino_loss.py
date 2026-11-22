from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from lightly.loss import emp_ssl_loss
from lightly.utils.dist import gather


class SimDINOLoss(Module):
    """Implementation of the loss from 'Simplifying DINO via Coding Rate
    Regularization'. [0]

    SimDINO keeps the student-teacher setup of DINO but replaces the prototype
    layer, centering, and temperature sharpening with two terms on the
    L2-normalized head outputs:

    - a compression term: the average cosine similarity between every student view
      and every teacher view, except a view with itself.
    - an expansion term: the coding rate of the global view features, which
      prevents collapse.

    The loss is ``-compression_coeff * compression - expansion``. This
    implementation follows the code published by the authors. [1]

    - [0]: SimDINO, 2025, https://arxiv.org/abs/2502.10385
    - [1]: https://github.com/RobinWu218/SimDINO

    Attributes:
        eps:
            Distortion epsilon of the coding rate. The reference implementation uses
            0.5.
        compression_coeff:
            Weight of the compression term.
        gather_distributed:
            If True, the global view features from all processes are gathered
            before the coding rate is computed, so it is estimated on the full
            batch. Otherwise each process uses its local batch.

    Examples:
        >>> # initialize loss function
        >>> loss_fn = SimDINOLoss()
        >>>
        >>> # generate two global views and any number of local views
        >>> global_views = [global_transform(images) for _ in range(2)]
        >>> local_views = [local_transform(images) for _ in range(6)]
        >>>
        >>> # the teacher only embeds the global views, the student embeds all views
        >>> # with the global views first
        >>> teacher_out = [teacher(view) for view in global_views]
        >>> student_out = [student(view) for view in global_views + local_views]
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(teacher_out, student_out)
    """

    def __init__(
        self,
        eps: float = 0.5,
        compression_coeff: float = 1.0,
        gather_distributed: bool = False,
    ) -> None:
        """Initializes the SimDINOLoss module.

        Raises:
            ValueError: If eps is not positive.
            ValueError: If gather_distributed is True but torch.distributed is not
                available.
        """
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )
        self.eps = eps
        self.compression_coeff = compression_coeff
        self.gather_distributed = gather_distributed

    def forward(self, teacher_out: list[Tensor], student_out: list[Tensor]) -> Tensor:
        """Computes the SimDINO loss.

        Args:
            teacher_out:
                List of tensors with shape (batch_size, dim) containing the
                L2-normalized teacher head outputs of the global views.
            student_out:
                List of tensors with shape (batch_size, dim) containing the
                L2-normalized student head outputs of all views. The global views
                must come first, in the same order as in teacher_out.

        Returns:
            The SimDINO loss.

        Raises:
            ValueError: If there are more teacher views than student views or if
                there are no cross-view terms for the compression term.
        """
        num_teacher_views = len(teacher_out)
        num_student_views = len(student_out)
        if num_teacher_views > num_student_views:
            raise ValueError(
                "SimDINOLoss expects the student to embed at least the teacher's "
                f"views, but got {num_teacher_views} teacher view(s) and "
                f"{num_student_views} student view(s)."
            )
        n_terms = num_teacher_views * num_student_views - num_teacher_views
        if n_terms == 0:
            raise ValueError(
                "SimDINOLoss requires at least two views in total (a view is not "
                f"compared with itself), but got {num_teacher_views} teacher view(s) "
                f"and {num_student_views} student view(s)."
            )

        teacher = torch.stack(teacher_out)
        student = torch.stack(student_out)

        # Compression: mean cosine similarity over the batch for every
        # (teacher view, student view) pair, skipping a view with itself.
        similarity = F.cosine_similarity(
            teacher.unsqueeze(1), student.unsqueeze(0), dim=-1
        ).mean(dim=-1)
        similarity = similarity * (
            1 - torch.eye(*similarity.shape, device=similarity.device)
        )
        compression = similarity.sum() / n_terms

        # Expansion: coding rate of the global views, using the average of the
        # student and teacher features as in the reference implementation.
        global_views = (student[:num_teacher_views] + teacher) / 2
        if self.gather_distributed and dist.is_initialized():
            if dist.get_world_size() > 1:
                global_views = torch.cat(gather(global_views), dim=1)
        expansion = -emp_ssl_loss.tcr_loss(global_views, eps=self.eps)
        # Balancing factor gamma from the reference implementation.
        _, batch_size, dim = global_views.shape
        expansion = expansion * (dim + batch_size) / (dim * batch_size)

        loss: Tensor = -self.compression_coeff * compression - expansion
        return loss
