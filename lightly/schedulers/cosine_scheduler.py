import warnings
from typing import Optional

import numpy as np

from lightly.schedulers.scheduler import Scheduler


class CosineWarmupScheduler(Scheduler):
    def __init__(
        self,
        max_steps: int,
        start_value: float = 1.0,
        end_value: float = 0.0,
        warmup_steps: int = 0,
        warmup_start_value: float = 0.0,
        warmup_end_value: Optional[float] = None,
        period: Optional[int] = None,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.start_value = start_value
        self.end_value = end_value
        self.warmup_steps = warmup_steps
        self.warmup_start_value = warmup_start_value
        self.warmup_end_value = warmup_end_value
        self.period = period

    def get_value(self, step: int) -> float:
        return cosine_warmup_schedule(
            step=step,
            max_steps=self.max_steps,
            start_value=self.start_value,
            end_value=self.end_value,
            warmup_steps=self.warmup_steps,
            warmup_start_value=self.warmup_start_value,
            warmup_end_value=self.warmup_end_value,
            period=self.period,
        )


def cosine_schedule(
    step: int,
    max_steps: int,
    start_value: float,
    end_value: float,
    period: Optional[int] = None,
) -> float:
    """Use cosine decay to gradually modify start_value to reach target end_value during
    iterations.

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
            If not provided, it defaults to max_steps.

    Returns:
        Cosine decay value.

    """
    if step < 0:
        raise ValueError("Current step number can't be negative")
    if max_steps < 1:
        raise ValueError("Total step number must be >= 1")
    if period is None and step > max_steps:
        warnings.warn(
            f"Current step number {step} exceeds max_steps {max_steps}.",
            category=RuntimeWarning,
        )
    if period is not None and period <= 0:
        raise ValueError("Period must be >= 1")

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
            Target value for warmup. If not provided, it defaults to start_value.

    Returns:
        Cosine decay value.

    """
    if warmup_steps < 0:
        raise ValueError("Warmup steps can't be negative")
    if warmup_steps > max_steps:
        raise ValueError("Warmup steps must be <= max_steps")
    if step > max_steps:
        warnings.warn(
            f"Current step number {step} exceeds max_steps {max_steps}.",
            category=RuntimeWarning,
        )

    if warmup_end_value is None:
        warmup_end_value = start_value

    if step < warmup_steps:
        return (
            warmup_start_value
            + (warmup_end_value - warmup_start_value) * (step + 1) / warmup_steps
        )
    elif period is not None:
        return cosine_schedule(
            step=step - warmup_steps,
            max_steps=1,
            start_value=start_value,
            end_value=end_value,
            period=period,
        )
    else:
        return cosine_schedule(
            step=step - warmup_steps,
            max_steps=max_steps - warmup_steps,
            start_value=start_value,
            end_value=end_value,
        )
