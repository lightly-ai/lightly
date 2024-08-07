from typing import Optional

import pytest

from lightly.schedulers import cosine_scheduler
from lightly.schedulers.cosine_scheduler import CosineWarmupScheduler


@pytest.mark.parametrize(
    "step, max_steps, start_value, end_value, period, expected",
    [
        # No period
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
    assert cosine_scheduler.cosine_schedule(
        step=step,
        max_steps=max_steps,
        start_value=start_value,
        end_value=end_value,
        period=period,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "step, max_steps, start_value, end_value, period, expected_message",
    [
        (-1, 1, 0.0, 1.0, None, "Current step number can't be negative"),
        (0, 0, 0.0, 1.0, None, "Total step number must be >= 1"),
        (1, 0, 0.0, 1.0, None, "Total step number must be >= 1"),
        (0, 1, 0.0, 1.0, -1, "Period must be >= 1"),
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
        cosine_scheduler.cosine_schedule(
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
        cosine_scheduler.cosine_schedule(
            step=11, max_steps=10, start_value=0.0, end_value=1.0
        )


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
    # No warmup, period
    (0, 20, 1.0, 0.0, 0, 0.0, None, 10, 1.0),
    (1, 20, 1.0, 0.0, 0, 0.0, None, 10, 0.90450849),
    (5, 20, 1.0, 0.0, 0, 0.0, None, 10, 0.0),
    (10, 20, 1.0, 0.0, 0, 0.0, None, 10, 1.0),
    (11, 20, 1.0, 0.0, 0, 0.0, None, 10, 0.90450849),
    (15, 20, 1.0, 0.0, 0, 0.0, None, 10, 0.0),
    (20, 20, 1.0, 0.0, 0, 0.0, None, 10, 1.0),
    # Warmup and period
    (0, 20, 1.0, 0.0, 10, 0.0, 0.5, 10, 0.05),
    (1, 20, 1.0, 0.0, 10, 0.0, 0.5, 10, 0.1),
    (2, 20, 1.0, 0.0, 10, 0.0, 0.5, 10, 0.15),
    (10, 20, 1.0, 0.0, 10, 0.0, 0.5, 10, 1.0),
    (11, 20, 1.0, 0.0, 0, 0.0, 0.5, 10, 0.90450849),
    (15, 20, 1.0, 0.0, 0, 0.0, 0.5, 10, 0.0),
    (20, 20, 1.0, 0.0, 0, 0.0, 0.5, 10, 1.0),
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
    assert cosine_scheduler.cosine_warmup_schedule(
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
        (-1, "Warmup steps can't be negative"),
        (2, "Warmup steps must be <= max_steps"),
    ],
)
def test_cosine_warmup_schedule__warmup_steps_error(
    warmup_steps: int,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        cosine_scheduler.cosine_warmup_schedule(
            step=0,
            max_steps=1,
            start_value=0.0,
            end_value=1.0,
            warmup_steps=warmup_steps,
            warmup_start_value=0.0,
        )


def test_cosine_warmup_schedule__error_warmup_greater_than_max_steps() -> None:
    with pytest.raises(ValueError, match="Warmup steps must be <= max_steps"):
        cosine_scheduler.cosine_warmup_schedule(
            step=0,
            max_steps=1,
            start_value=0.0,
            end_value=1.0,
            warmup_steps=2,
            warmup_start_value=0.0,
        )


def test_cosine_warmup_schedule__warn_step_exceeds_max_steps() -> None:
    with pytest.warns(
        RuntimeWarning, match="Current step number 11 exceeds max_steps 10."
    ):
        cosine_scheduler.cosine_warmup_schedule(
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
        scheduler = CosineWarmupScheduler(
            max_steps=max_steps,
            start_value=start_value,
            end_value=end_value,
            warmup_steps=warmup_steps,
            warmup_start_value=warmup_start_value,
            warmup_end_value=warmup_end_value,
            period=period,
        )
        assert scheduler.get_value(step=step) == pytest.approx(expected)
