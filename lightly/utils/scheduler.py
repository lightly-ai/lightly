import warnings
from typing import Optional

import numpy as np
import torch


def cosine_schedule(
    step: int,
    max_steps: int,
    start_value: float,
    end_value: float,
    period: Optional[int] = None,
) -> float:
    """Use cosine decay to gradually modify start_value to reach target end_value.

    Args:
        step:
            Current step number.
        max_steps:
            Total number of steps.
        start_value:
            Starting value.
        end_value:
            Target value.
        period:
            The number of steps over which the cosine function completes a full cycle.
            Defaults to max_steps.

    Returns:
        Cosine decay value.

    """
    if step < 0:
        raise ValueError(f"Current step number {step} can't be negative.")
    if max_steps < 1:
        raise ValueError(f"Total step number {max_steps} must be >= 1.")
    if period is None and step > max_steps:
        warnings.warn(
            f"Current step number {step} exceeds max_steps {max_steps}.",
            category=RuntimeWarning,
        )
    if period is not None and period <= 0:
        raise ValueError(f"Period {period} must be >= 1")

    decay: float
    if period is not None:  # "cycle" based on period, if provided
        decay = (
            end_value
            - (end_value - start_value) * (np.cos(2 * np.pi * step / period) + 1) / 2
        )
    elif max_steps == 1:
        # Avoid division by zero
        decay = end_value
    elif step == max_steps:
        # Special case for Pytorch Lightning which updates LR scheduler also for epoch
        # after last training epoch.
        decay = end_value
    else:
        decay = (
            end_value
            - (end_value - start_value)
            * (np.cos(np.pi * step / (max_steps - 1)) + 1)
            / 2
        )
    return decay


def cosine_warmup_schedule(
    step: int,
    max_steps: int,
    start_value: float,
    end_value: float,
    warmup_steps: int,
    warmup_start_value: float,
    warmup_end_value: Optional[float] = None,
    period: Optional[int] = None,
) -> float:
    """Use cosine decay to gradually modify start_value to reach target end_value.

    Uses linear warmup for the first warmup_steps steps.

    Args:
        step:
            Current step number.
        max_steps:
            Total number of steps.
        start_value:
            Starting value.
        end_value:
            Target value.
        warmup_steps:
            Number of steps for warmup.
        warmup_start_value:
            Starting value for warmup.
        warmup_end_value:
            Target value for warmup. Defaults to start_value.
        period:
            The number of steps over which the cosine function completes a full cycle.
            Defaults to max_steps - warmup_steps.
    Returns:
        Cosine decay value.
    """
    if warmup_steps < 0:
        raise ValueError(f"Warmup steps {warmup_steps} can't be negative.")
    if warmup_steps > max_steps:
        raise ValueError(f"Warmup steps {warmup_steps} must be <= max_steps.")
    if step > max_steps:
        warnings.warn(
            f"Current step number {step} exceeds max_steps {max_steps}.",
            category=RuntimeWarning,
        )

    if warmup_end_value is None:
        warmup_end_value = start_value

    if step < warmup_steps:
        # Use step + 1 to reach warmup_end_value at end of warmup. This means that the
        # initial warmup_start_value is skipped which is oftentimes desired when setting
        # it to 0 as this would result in no parameter updates.
        return (
            warmup_start_value
            + (warmup_end_value - warmup_start_value) * (step + 1) / warmup_steps
        )
    else:
        max_steps = max_steps - warmup_steps if period is None else 1
        return cosine_schedule(
            step=step - warmup_steps,
            max_steps=max_steps,
            start_value=start_value,
            end_value=end_value,
            period=period,
        )


class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Cosine warmup scheduler for learning rate.

    Args:
        optimizer:
            Optimizer object to schedule the learning rate.
        warmup_epochs:
            Number of warmup epochs or steps.
        max_epochs:
            Total number of training epochs or steps.
        last_epoch:
            The index of last epoch or step.
        start_value:
            Starting learning rate.
        end_value:
            Target learning rate.
        verbose:
            If True, prints a message to stdout for each update.
        warmup_start_value:
            Starting learning rate for warmup.
        warmup_end_value:
            Target learning rate for warmup. Defaults to start_value.

    Note: The `epoch` arguments do not necessarily have to be epochs. Any step or index
    can be used. The naming follows the PyTorch convention to use `epoch` for the steps
    in the scheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        last_epoch: int = -1,
        start_value: float = 1.0,
        end_value: float = 0.001,
        period: Optional[int] = None,
        verbose: bool = False,
        warmup_start_value: float = 0.0,
        warmup_end_value: Optional[float] = None,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.start_value = start_value
        self.end_value = end_value
        self.period = period
        self.warmup_start_value = warmup_start_value
        self.warmup_end_value = warmup_end_value

        super().__init__(
            optimizer=optimizer,
            lr_lambda=self.scale_lr,
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def scale_lr(self, epoch: int) -> float:
        """Scale learning rate according to the current epoch number.

        Args:
            epoch:
                Current epoch number.

        Returns:
            Scaled learning rate.

        """
        return cosine_warmup_schedule(
            step=epoch,
            max_steps=self.max_epochs,
            start_value=self.start_value,
            end_value=self.end_value,
            warmup_steps=self.warmup_epochs,
            warmup_start_value=self.warmup_start_value,
            warmup_end_value=self.warmup_end_value,
            period=self.period,
        )
