from contextlib import contextmanager
from typing import Generator

import pytest
import torch

import lightly.models.modules.sparse_spark as sparse_spark


@contextmanager
def _cleanup_curr_active() -> Generator[None, None, None]:
    try:
        yield
    finally:
        sparse_spark._cur_active = None


def test__get_active_ex_or_ii_expands_mask() -> None:
    with _cleanup_curr_active():
        H, W = 32, 32
        sparse_spark._cur_active = torch.tensor(
            [
                [
                    [
                        [1, 0],
                        [0, 1],
                    ]
                ]
            ],
            dtype=torch.bool,
        )

        active = sparse_spark._get_active_ex_or_ii(H=H, W=W, returning_active_ex=True)
        assert not isinstance(active, tuple)

        assert active.shape == (1, 1, H, W)
        assert active[:, :, :16, :16].all()
        assert active[:, :, :16, 16:].logical_not().all()
        assert active[:, :, 16:, :16].logical_not().all()
        assert active[:, :, 16:, 16:].all()


def test__get_active_ex_or_ii_dont_shrink_mask() -> None:
    with _cleanup_curr_active():
        H, W = 4, 4
        sparse_spark._cur_active = torch.ones(1, 1, 32, 32)
        with pytest.raises(AssertionError):
            sparse_spark._get_active_ex_or_ii(H=H, W=W, returning_active_ex=False)


def test__get_active_ex_or_ii_raise_on_non_active_mask() -> None:
    with _cleanup_curr_active():
        H, W = 32, 32
        sparse_spark._cur_active = None
        with pytest.raises(
            AssertionError,
        ):
            sparse_spark._get_active_ex_or_ii(H=H, W=W, returning_active_ex=False)


def test__get_active_ex_or_ii_returning_ex_false_correct_values() -> None:
    with _cleanup_curr_active():
        H, W = 32, 32
        sparse_spark._cur_active = torch.tensor(
            [
                [
                    [
                        [1, 0],
                        [0, 1],
                    ]
                ]
            ],
            dtype=torch.bool,
        )

        active_b, active_h, active_w = sparse_spark._get_active_ex_or_ii(
            H=H, W=W, returning_active_ex=False
        )
        active_ex = sparse_spark._get_active_ex_or_ii(
            H=H, W=W, returning_active_ex=True
        )
        assert not isinstance(active_ex, tuple)

        active_ex_scattered = torch.zeros_like(active_ex)
        active_ex_scattered[active_b, :, active_h, active_w] = 1

        assert torch.equal(active_ex, active_ex_scattered)
