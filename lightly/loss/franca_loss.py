"""Franca loss.

- [0]: Franca, 2025, https://arxiv.org/abs/2507.14137
- [1]: https://github.com/valeoai/Franca
"""

from __future__ import annotations

from typing import List, Sequence, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList

from lightly.loss import capi_loss
from lightly.models.modules import center
from lightly.models.modules.center import Center

CENTER_MODES = ("mean", "sinkhorn")


class FrancaDINOLoss(Module):
    """Matryoshka DINO loss used by Franca. [0]

    Franca applies DINO clustering at several nested granularity levels at once,
    following Matryoshka Representation Learning. This loss computes the standard DINO
    image-level cross-entropy [1] for every level and sums the level losses without
    weighting, matching the reference implementation [2] (which stores a per-level
    ``relative_importance`` but does not apply it in the loss).

    Two teacher-centering modes are supported, both present in the reference: ``"mean"``
    subtracts a per-level running center before the softmax (the DINO/DINOv2 way), and
    ``"sinkhorn"`` turns the teacher logits into soft targets with the Sinkhorn-Knopp
    normalization reused from CAPI [3]. The per-level cross-entropy follows lightly's
    :class:`~lightly.loss.dino_loss.DINOLoss`, so a single-level loss is equivalent to it.

    - [0]: Franca, 2025, https://arxiv.org/abs/2507.14137
    - [1]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [2]: https://github.com/valeoai/Franca
    - [3]: CAPI, 2025, https://arxiv.org/abs/2502.08769

    Attributes:
        output_dims:
            Number of prototypes per nesting level. Must match the per-level output
            dimensions of the Franca projection head.
        center_mode:
            Teacher-centering mode, one of ``"mean"`` or ``"sinkhorn"``.

    Examples:
        >>> loss_fn = FrancaDINOLoss(output_dims=[32768, 65536])
        >>> # each view is a tuple with one head output per nesting level
        >>> teacher_out = [teacher(view0), teacher(view1)]
        >>> student_out = [student(view0), student(view1)]
        >>> loss = loss_fn(teacher_out, student_out)
    """

    def __init__(
        self,
        output_dims: Sequence[int],
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        center_mode: str = "mean",
        sinkhorn_iterations: int = 3,
        gather_distributed: bool = False,
    ) -> None:
        """Initializes the FrancaDINOLoss module.

        Args:
            output_dims:
                Number of prototypes per nesting level, matching the projection head.
            warmup_teacher_temp:
                Initial teacher temperature for the warmup schedule.
            teacher_temp:
                Final teacher temperature after warmup.
            warmup_teacher_temp_epochs:
                Number of epochs of the teacher-temperature warmup schedule.
            student_temp:
                Temperature applied to the student outputs.
            center_momentum:
                Momentum for the running center in ``"mean"`` mode.
            center_mode:
                Teacher-centering mode, one of ``"mean"`` or ``"sinkhorn"``.
            sinkhorn_iterations:
                Number of Sinkhorn-Knopp iterations in ``"sinkhorn"`` mode.
            gather_distributed:
                If True, the Sinkhorn normalization is synchronized across processes.

        Raises:
            ValueError: If ``output_dims`` is empty or has a non-positive entry, if
                ``center_mode`` is unknown, or if ``gather_distributed`` is True while
                torch.distributed is not available.
        """
        super().__init__()
        dims = list(output_dims)
        if not dims:
            raise ValueError("output_dims must not be empty.")
        if any(dim <= 0 for dim in dims):
            raise ValueError(f"output_dims must be positive, got {dims}.")
        if center_mode not in CENTER_MODES:
            raise ValueError(
                f"Unknown center_mode '{center_mode}'. Valid modes are {list(CENTER_MODES)}."
            )
        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or use a distributed-enabled "
                "installation of PyTorch."
            )
        self.output_dims = dims
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_mode = center_mode
        self.sinkhorn_iterations = sinkhorn_iterations
        self.gather_distributed = gather_distributed

        # One teacher center per nesting level, each of the level's own width.
        self._center_names: List[str] = []
        for level, dim in enumerate(dims):
            name = f"center_{level}"
            self.register_buffer(name, torch.zeros(1, 1, dim))
            self._center_names.append(name)

        # Linear warmup schedule for the teacher temperature (mirrors DINOLoss).
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp_schedule = torch.linspace(
            start=warmup_teacher_temp,
            end=teacher_temp,
            steps=warmup_teacher_temp_epochs,
        )

    def forward(
        self,
        teacher_out: Sequence[Sequence[Tensor]],
        student_out: Sequence[Sequence[Tensor]],
        teacher_temp: float | None = None,
        epoch: int | None = None,
    ) -> Tensor:
        """Sums the per-level DINO cross-entropy over the nesting levels.

        Args:
            teacher_out:
                One entry per teacher view. Each entry is a sequence with one tensor of
                shape (batch_size, output_dims[level]) per nesting level.
            student_out:
                One entry per student view, shaped like ``teacher_out``.
            teacher_temp:
                Teacher temperature. If None, the schedule (with ``epoch``) or the
                default temperature is used.
            epoch:
                Current epoch, used to read the warmup schedule when ``teacher_temp``
                is None.

        Returns:
            The summed cross-entropy loss over all nesting levels.

        Raises:
            ValueError: If a view does not provide one output per nesting level, or if
                fewer than two views in total are given.
        """
        num_levels = len(self.output_dims)
        teacher_views = list(teacher_out)
        student_views = list(student_out)
        if not teacher_views or not student_views:
            raise ValueError(
                "teacher_out and student_out must each contain at least one view."
            )
        for view in teacher_views + student_views:
            if len(view) != num_levels:
                raise ValueError(
                    f"Every view must provide one output per nesting level "
                    f"({num_levels}), but got a view with {len(view)}."
                )

        teacher_temperature = self._teacher_temperature(
            teacher_temp=teacher_temp, epoch=epoch
        )

        loss = teacher_views[0][0].new_zeros(())
        for level in range(num_levels):
            teacher_level = [view[level] for view in teacher_views]
            student_level = [view[level] for view in student_views]
            loss = loss + self._level_loss(
                level=level,
                teacher_level=teacher_level,
                student_level=student_level,
                teacher_temperature=teacher_temperature,
            )
        return loss

    def _teacher_temperature(
        self, teacher_temp: float | None, epoch: int | None
    ) -> Tensor:
        """Resolves the teacher temperature, mirroring DINOLoss."""
        if teacher_temp is not None:
            return torch.tensor(teacher_temp)
        if epoch is not None:
            if epoch < self.warmup_teacher_temp_epochs:
                return self.teacher_temp_schedule[epoch]
            return torch.tensor(self.teacher_temp)
        return torch.tensor(self.teacher_temp)

    def _level_loss(
        self,
        level: int,
        teacher_level: List[Tensor],
        student_level: List[Tensor],
        teacher_temperature: Tensor,
    ) -> Tensor:
        """Standard DINO cross-entropy for one nesting level."""
        teacher_stacked = torch.stack(teacher_level)
        student_stacked = torch.stack(student_level)
        teacher_targets = self._teacher_targets(
            level=level,
            teacher_stacked=teacher_stacked,
            teacher_temperature=teacher_temperature,
        )
        student_log = F.log_softmax(student_stacked / self.student_temp, dim=-1)

        # b = batch_size, t = teacher views, s = student views, d = clusters.
        loss = -torch.einsum("tbd,sbd->ts", teacher_targets, student_log)
        loss.fill_diagonal_(0)

        n_terms = loss.numel() - loss.diagonal().numel()
        if n_terms == 0:
            raise ValueError(
                "FrancaDINOLoss requires at least two views in total (the diagonal "
                "matching the same view index is excluded), but got "
                f"{len(teacher_level)} teacher view(s) and {len(student_level)} "
                "student view(s), which leaves no cross-view terms to compute the "
                "loss from."
            )
        batch_size = teacher_stacked.shape[1]
        return loss.sum() / (n_terms * batch_size)

    def _teacher_targets(
        self, level: int, teacher_stacked: Tensor, teacher_temperature: Tensor
    ) -> Tensor:
        """Produces the teacher targets for one level with the configured centering."""
        if self.center_mode == "sinkhorn":
            n_views, batch_size, dim = teacher_stacked.shape
            # Treat every teacher token as a length-1 sequence so the Sinkhorn-Knopp
            # balancing runs over the batch of all teacher-view samples of this level.
            logits = teacher_stacked.reshape(n_views * batch_size, 1, dim)
            assignments = capi_loss.sinkhorn_knopp(
                logits=logits / teacher_temperature,
                iterations=self.sinkhorn_iterations,
                gather_distributed=self.gather_distributed,
            )
            return assignments.reshape(n_views, batch_size, dim)

        name = self._center_names[level]
        center_buffer = getattr(self, name)
        targets = F.softmax(
            (teacher_stacked - center_buffer) / teacher_temperature, dim=-1
        )
        self._update_center(name=name, teacher_stacked=teacher_stacked)
        return targets

    @torch.no_grad()
    def _update_center(self, name: str, teacher_stacked: Tensor) -> None:
        """Moving-average update of the running center for one level."""
        batch_center = center.center_mean(x=teacher_stacked, dim=(0, 1))
        getattr(self, name).data = center.center_momentum(
            center=getattr(self, name),
            batch_center=batch_center,
            momentum=self.center_momentum,
        )


class FrancaIBOTPatchLoss(Module):
    """Matryoshka iBOT patch loss used by Franca. [0]

    This is the masked-patch counterpart of :class:`FrancaDINOLoss`. It computes the
    standard iBOT patch cross-entropy [1] (as used in DINOv2 [2]) for every nesting
    level and sums the level losses without weighting, matching the reference
    implementation [3]. The per-level cross-entropy follows lightly's
    :class:`~lightly.loss.ibot_loss.IBOTPatchLoss`, so a single-level loss is equivalent
    to it, including the per-image masked-token weighting.

    Two teacher-centering modes are supported: ``"mean"`` keeps a per-level running
    center (reusing :class:`~lightly.models.modules.center.Center`), and ``"sinkhorn"``
    turns the teacher logits into soft targets with the Sinkhorn-Knopp normalization
    reused from CAPI [4].

    - [0]: Franca, 2025, https://arxiv.org/abs/2507.14137
    - [1]: iBOT, 2021, https://arxiv.org/abs/2111.07832
    - [2]: DINOv2, 2023, https://arxiv.org/abs/2304.07193
    - [3]: https://github.com/valeoai/Franca
    - [4]: CAPI, 2025, https://arxiv.org/abs/2502.08769

    Attributes:
        output_dims:
            Number of prototypes per nesting level, matching the projection head.
        center_mode:
            Teacher-centering mode, one of ``"mean"`` or ``"sinkhorn"``.
        centers:
            One :class:`~lightly.models.modules.center.Center` per level in ``"mean"``
            mode; empty in ``"sinkhorn"`` mode.
    """

    def __init__(
        self,
        output_dims: Sequence[int],
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_mode: str = "mean",
        center_momentum: float = 0.9,
        sinkhorn_iterations: int = 3,
        gather_distributed: bool = False,
    ) -> None:
        """Initializes the FrancaIBOTPatchLoss module.

        Args:
            output_dims:
                Number of prototypes per nesting level, matching the projection head.
            teacher_temp:
                Temperature applied to the teacher outputs.
            student_temp:
                Temperature applied to the student outputs.
            center_mode:
                Teacher-centering mode, one of ``"mean"`` or ``"sinkhorn"``.
            center_momentum:
                Momentum for the running center in ``"mean"`` mode.
            sinkhorn_iterations:
                Number of Sinkhorn-Knopp iterations in ``"sinkhorn"`` mode.
            gather_distributed:
                If True, the Sinkhorn normalization is synchronized across processes.

        Raises:
            ValueError: If ``output_dims`` is empty or has a non-positive entry, if
                ``center_mode`` is unknown, or if ``gather_distributed`` is True while
                torch.distributed is not available.
        """
        super().__init__()
        dims = list(output_dims)
        if not dims:
            raise ValueError("output_dims must not be empty.")
        if any(dim <= 0 for dim in dims):
            raise ValueError(f"output_dims must be positive, got {dims}.")
        if center_mode not in CENTER_MODES:
            raise ValueError(
                f"Unknown center_mode '{center_mode}'. Valid modes are {list(CENTER_MODES)}."
            )
        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or use a distributed-enabled "
                "installation of PyTorch."
            )
        self.output_dims = dims
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_mode = center_mode
        self.sinkhorn_iterations = sinkhorn_iterations
        self.gather_distributed = gather_distributed

        # One center per level in mean mode; sinkhorn mode keeps no running center.
        if center_mode == "mean":
            self.centers = ModuleList(
                Center(size=(1, dim), mode="mean", momentum=center_momentum)
                for dim in dims
            )
        else:
            self.centers = ModuleList()

    def forward(
        self,
        teacher_out: Sequence[Tensor],
        student_out: Sequence[Tensor],
        mask: Tensor,
        teacher_temp: float | None = None,
    ) -> Tensor:
        """Sums the per-level iBOT patch cross-entropy over the nesting levels.

        Args:
            teacher_out:
                One tensor per nesting level, each of shape
                (batch_size * num_masked_tokens, output_dims[level]) containing the
                teacher output of the masked tokens.
            student_out:
                One tensor per nesting level, shaped like ``teacher_out``, from the
                student.
            mask:
                Boolean tensor of shape (batch_size, height, width) marking the masked
                tokens. The number of True entries must equal the number of rows of each
                level tensor, in the same row-major order.
            teacher_temp:
                Teacher temperature. If None, the default temperature is used.

        Returns:
            The summed iBOT patch loss over all nesting levels.

        Raises:
            ValueError: If ``teacher_out`` or ``student_out`` does not provide one
                tensor per nesting level, or if ``mask`` marks no masked tokens.
        """
        num_levels = len(self.output_dims)
        teacher_levels = list(teacher_out)
        student_levels = list(student_out)
        if len(teacher_levels) != num_levels or len(student_levels) != num_levels:
            raise ValueError(
                f"teacher_out and student_out must each provide one tensor per nesting "
                f"level ({num_levels}), but got {len(teacher_levels)} and "
                f"{len(student_levels)}."
            )
        # An all-false mask leaves the teacher tensors empty, which makes the center
        # update and the loss NaN. Reject it with a clear error instead.
        if not bool(mask.any()):
            raise ValueError("mask must mark at least one masked token.")

        teacher_temperature = torch.tensor(
            teacher_temp if teacher_temp is not None else self.teacher_temp
        )

        # Per-image weight of each masked token, shared across levels.
        num_masked_per_image = mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        weight = (1.0 / num_masked_per_image).expand_as(mask)[mask]
        batch_size = mask.shape[0]

        loss = teacher_levels[0].new_zeros(())
        for level in range(num_levels):
            loss = loss + self._level_loss(
                level=level,
                teacher_level=teacher_levels[level],
                student_level=student_levels[level],
                weight=weight,
                batch_size=batch_size,
                teacher_temperature=teacher_temperature,
            )
        return loss

    def _level_loss(
        self,
        level: int,
        teacher_level: Tensor,
        student_level: Tensor,
        weight: Tensor,
        batch_size: int,
        teacher_temperature: Tensor,
    ) -> Tensor:
        """Standard iBOT patch cross-entropy for one nesting level."""
        teacher_targets = self._teacher_targets(
            level=level,
            teacher_level=teacher_level,
            teacher_temperature=teacher_temperature,
        )
        student_log = F.log_softmax(student_level / self.student_temp, dim=-1)
        cross_entropy = -torch.sum(teacher_targets * student_log, dim=-1)
        return (cross_entropy * weight).sum() / batch_size

    def _teacher_targets(
        self, level: int, teacher_level: Tensor, teacher_temperature: Tensor
    ) -> Tensor:
        """Produces the teacher targets for one level with the configured centering."""
        if self.center_mode == "sinkhorn":
            n_tokens, dim = teacher_level.shape
            # Each masked token is a length-1 sequence, so Sinkhorn-Knopp balances the
            # assignments over the batch of all masked tokens of this level.
            logits = teacher_level.reshape(n_tokens, 1, dim)
            assignments = capi_loss.sinkhorn_knopp(
                logits=logits / teacher_temperature,
                iterations=self.sinkhorn_iterations,
                gather_distributed=self.gather_distributed,
            )
            return assignments.reshape(n_tokens, dim)

        center_module = cast(Center, self.centers[level])
        targets = F.softmax(
            (teacher_level - center_module.value) / teacher_temperature, dim=-1
        )
        center_module.update(teacher_level)
        return targets
