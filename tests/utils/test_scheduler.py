import unittest
from typing import Optional

import pytest
import torch
from torch import nn
from torch.nn import Linear
from torch.optim import SGD

from lightly.utils import scheduler
from lightly.utils.scheduler import CosineWarmupScheduler


@pytest.mark.parametrize(
    "step, max_steps, start_value, end_value, period, expected",
    [
        # No period
        (0, 0, 1.0, 0.0, None, 0.0),
        (1, 0, 1.0, 0.0, None, 0.0),
        (1, 1, 1.0, 0.0, None, 0.0),
        (0, 10, 1.0, 0.0, None, 1.0),
        (1, 10, 1.0, 0.0, None, 0.96984631),
        (2, 10, 1.0, 0.0, None, 0.88302222),
        (10, 10, 1.0, 0.0, None, 0.0),
        # Period
        (0, 20, 1.0, 0.0, 10, 1.0),
        (1, 20, 1.0, 0.0, 10, 0.90450849),
        (5, 20, 1.0, 0.0, 10, 0.0),
        (10, 20, 1.0, 0.0, 10, 1.0),
        (11, 20, 1.0, 0.0, 10, 0.90450849),
        (15, 20, 1.0, 0.0, 10, 0.0),
        (20, 20, 1.0, 0.0, 10, 1.0),
        (20, 20, 1.0, 0.0, 20, 1.0),
    ],
)
def test_cosine_schedule(
    step: int,
    max_steps: int,
    start_value: float,
    end_value: float,
    period: Optional[int],
    expected: float,
) -> None:
    assert scheduler.cosine_schedule(
        step=step,
        max_steps=max_steps,
        start_value=start_value,
        end_value=end_value,
        period=period,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "step, max_steps, start_value, end_value, period, expected_message",
    [
        (-1, 1, 0.0, 1.0, None, "Current step number -1 can't be negative"),
        (1, -1, 0.0, 1.0, None, "Total step number -1 can't be negative"),
        (0, 1, 0.0, 1.0, -1, "Period -1 must be >= 1"),
    ],
)
def test_cosine_schedule__error(
    step: int,
    max_steps: int,
    start_value: float,
    end_value: float,
    period: Optional[int],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        scheduler.cosine_schedule(
            step=step,
            max_steps=max_steps,
            start_value=start_value,
            end_value=end_value,
            period=period,
        )


def test_cosine_schedule__warn_step_exceeds_max_steps() -> None:
    with pytest.warns(
        RuntimeWarning, match="Current step number 11 exceeds max_steps 10."
    ):
        scheduler.cosine_schedule(step=11, max_steps=10, start_value=0.0, end_value=1.0)


COSINE_WARMUP_VALUES = [
    # No warmup, no period
    (0, 10, 1.0, 0.0, 0, 0.0, None, None, 1.0),
    (1, 10, 1.0, 0.0, 0, 0.0, None, None, 0.96984631),
    (2, 10, 1.0, 0.0, 0, 0.0, None, None, 0.88302222),
    (10, 10, 1.0, 0.0, 0, 0.0, None, None, 0.0),
    # Warmup, no period
    (0, 20, 1.0, 0.0, 10, 0.0, 0.5, None, 0.05),
    (1, 20, 1.0, 0.0, 10, 0.0, 0.5, None, 0.1),
    (2, 20, 1.0, 0.0, 10, 0.0, 0.5, None, 0.15),
    (10, 20, 1.0, 0.0, 10, 0.0, 0.5, None, 1.0),
    (11, 20, 1.0, 0.0, 10, 0.0, 0.5, None, 0.96984631),
    (20, 20, 1.0, 0.0, 10, 0.0, 0.5, None, 0.0),
    (20, 20, 1.0, 0.0, 20, 0.0, 0.5, None, 0.0),
    # No warmup, period
    (0, 20, 1.0, 0.0, 0, 0.0, None, 10, 1.0),
    (1, 20, 1.0, 0.0, 0, 0.0, None, 10, 0.90450849),
    (5, 20, 1.0, 0.0, 0, 0.0, None, 10, 0.0),
    (10, 20, 1.0, 0.0, 0, 0.0, None, 10, 1.0),
    (11, 20, 1.0, 0.0, 0, 0.0, None, 10, 0.90450849),
    (15, 20, 1.0, 0.0, 0, 0.0, None, 10, 0.0),
    (20, 20, 1.0, 0.0, 0, 0.0, None, 10, 1.0),
    (20, 20, 1.0, 0.0, 0, 0.0, None, 20, 1.0),
    # Warmup and period
    (0, 20, 1.0, 0.0, 10, 0.0, 0.5, 10, 0.05),
    (1, 20, 1.0, 0.0, 10, 0.0, 0.5, 10, 0.1),
    (2, 20, 1.0, 0.0, 10, 0.0, 0.5, 10, 0.15),
    (10, 20, 1.0, 0.0, 10, 0.0, 0.5, 10, 1.0),
    (11, 20, 1.0, 0.0, 0, 0.0, 0.5, 10, 0.90450849),
    (15, 20, 1.0, 0.0, 0, 0.0, 0.5, 10, 0.0),
    (20, 20, 1.0, 0.0, 0, 0.0, 0.5, 10, 1.0),
    (20, 20, 1.0, 0.0, 20, 0.0, 0.5, 10, 1.0),
    (20, 20, 1.0, 0.0, 20, 0.0, 0.5, 20, 1.0),
]


@pytest.mark.parametrize(
    "step, max_steps, start_value, end_value, warmup_steps, warmup_start_value, warmup_end_value, period, expected",
    COSINE_WARMUP_VALUES,
)
def test_cosine_warmup_schedule(
    step: int,
    max_steps: int,
    start_value: float,
    end_value: float,
    warmup_steps: int,
    warmup_start_value: float,
    warmup_end_value: Optional[float],
    period: Optional[int],
    expected: float,
) -> None:
    assert scheduler.cosine_warmup_schedule(
        step=step,
        max_steps=max_steps,
        start_value=start_value,
        end_value=end_value,
        warmup_steps=warmup_steps,
        warmup_start_value=warmup_start_value,
        warmup_end_value=warmup_end_value,
        period=period,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "warmup_steps, expected_message",
    [
        (-1, "Warmup steps -1 can't be negative"),
        (2, "Warmup steps 2 must be <= max_steps"),
    ],
)
def test_cosine_warmup_schedule__error_warmup_steps(
    warmup_steps: int,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        scheduler.cosine_warmup_schedule(
            step=0,
            max_steps=1,
            start_value=0.0,
            end_value=1.0,
            warmup_steps=warmup_steps,
            warmup_start_value=0.0,
        )


def test_cosine_warmup_schedule__warn_step_exceeds_max_steps() -> None:
    with pytest.warns(
        RuntimeWarning, match="Current step number 11 exceeds max_steps 10."
    ):
        scheduler.cosine_warmup_schedule(
            step=11,
            max_steps=10,
            start_value=0.0,
            end_value=1.0,
            warmup_steps=0,
            warmup_start_value=0.0,
        )


class TestCosineWarmupScheduler:
    @pytest.mark.parametrize(
        "step, max_steps, start_value, end_value, warmup_steps, warmup_start_value, warmup_end_value, period, expected",
        COSINE_WARMUP_VALUES,
    )
    def test_get_value(
        self,
        step: int,
        max_steps: int,
        start_value: float,
        end_value: float,
        warmup_steps: int,
        warmup_start_value: float,
        warmup_end_value: float,
        period: Optional[int],
        expected: float,
    ) -> None:
        optimizer = SGD(Linear(1, 1).parameters(), lr=1.0)
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            max_epochs=max_steps,
            start_value=start_value,
            end_value=end_value,
            warmup_epochs=warmup_steps,
            warmup_start_value=warmup_start_value,
            warmup_end_value=warmup_end_value,
            period=period,
        )
        for _ in range(step):
            scheduler.step()
        assert scheduler.get_last_lr()[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    "step, warmup_steps, start_value, end_value, expected",
    [
        # start from 0.0 to 1.0
        (0, 10, 0.0, 1.0, 0.0),
        (5, 10, 0.0, 1.0, 0.5),
        (9, 10, 0.0, 1.0, 0.9),
        (10, 10, 0.0, 1.0, 1.0),
        (15, 10, 0.0, 1.0, 1.0),
        # start from 2.0 to 4.0
        (0, 10, 2.0, 4.0, 2.0),
        (5, 10, 2.0, 4.0, 3.0),
        (9, 10, 2.0, 4.0, 3.8),
        (10, 10, 2.0, 4.0, 4.0),
        (15, 10, 2.0, 4.0, 4.0),
        # start value = end value
        (0, 10, 2.0, 2.0, 2.0),
        (5, 10, 2.0, 2.0, 2.0),
        (9, 10, 2.0, 2.0, 2.0),
        (10, 10, 2.0, 2.0, 2.0),
        (15, 10, 2.0, 2.0, 2.0),
        # warmup step 0
        (0, 0, 2.0, 4.0, 4.0),
        (1, 0, 2.0, 4.0, 4.0),
        # warmup step 1
        (0, 1, 2.0, 4.0, 2.0),
        (1, 1, 2.0, 4.0, 4.0),
    ],
)
def test_linear_warmup_schedule(
    step: int,
    warmup_steps: int,
    start_value: float,
    end_value: float,
    expected: float,
) -> None:
    assert scheduler.linear_warmup_schedule(
        step=step,
        warmup_steps=warmup_steps,
        start_value=start_value,
        end_value=end_value,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "step, warmup_steps, start_value, end_value, expected_message",
    [
        (1, -1, 0.0, 1.0, "Warmup steps -1 can't be negative."),
        (-1, 1, 0.0, 1.0, "Current step number -1 can't be negative."),
        (0, 1, -1.0, 1.0, "Start value -1.0 can't be negative."),
        (0, 1, 0.0, 0.0, "End value 0.0 can't be non-positive."),
        (0, 1, 0.0, -1.0, "End value -1.0 can't be non-positive."),
        (
            0,
            1,
            2.0,
            1.0,
            "Start value 2.0 must be less than or equal to end value 1.0.",
        ),
    ],
)
def test_linear_warmup_schedule__error(
    step: int,
    warmup_steps: int,
    start_value: float,
    end_value: float,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        scheduler.linear_warmup_schedule(
            step=step,
            warmup_steps=warmup_steps,
            start_value=start_value,
            end_value=end_value,
        )
