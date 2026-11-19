from typing import List

import pytest
import torch
from torch import Tensor

from lightly.models.modules.center import Center
from tests.ddp_helpers import NUM_PROCESSES, USE_PYTEST_POOL


def _accumulate_worker(
    rank: int, world_size: int, chunks: List[List[Tensor]]
) -> Tensor:
    # Pool worker: accumulate this rank's chunks, then apply one update.
    center = Center(size=(1, 4), momentum=0.9)
    for chunk in chunks[rank]:
        center.accumulate(chunk)
    center.apply_update()
    return center.value


def _single_update_worker(
    rank: int, world_size: int, chunks: List[List[Tensor]]
) -> Tensor:
    # Pool worker: single update over this rank's concatenated chunks.
    center = Center(size=(1, 4), momentum=0.9)
    center.update(torch.cat(chunks[rank], dim=0))
    return center.value


class TestCenter:
    def test__init__invalid_mode(self) -> None:
        with pytest.raises(ValueError):
            Center(size=(1, 32), mode="invalid")

    def test_value(self) -> None:
        center = Center(size=(1, 32), mode="mean")
        assert torch.all(center.value == 0)

    @pytest.mark.parametrize(
        "x, expected",
        [
            (torch.tensor([[0.0, 0.0], [0.0, 0.0]]), torch.tensor([0.0, 0.0])),
            (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([2.0, 3.0])),
        ],
    )
    def test_update(self, x: Tensor, expected: Tensor) -> None:
        center = Center(size=(1, 2), mode="mean", momentum=0.0)
        center.update(x)
        assert torch.all(center.value == expected)

    @pytest.mark.parametrize(
        "momentum, expected",
        [
            (0.0, torch.tensor([1.0, 2.0])),
            (0.1, torch.tensor([0.9, 1.8])),
            (0.5, torch.tensor([0.5, 1.0])),
            (1.0, torch.tensor([0.0, 0.0])),
        ],
    )
    def test_update__momentum(self, momentum: float, expected: Tensor) -> None:
        center = Center(size=(1, 2), mode="mean", momentum=momentum)
        center.update(torch.tensor([[1.0, 2.0]]))
        assert torch.all(center.value == expected)

    def test_update__no_argument(self) -> None:
        """update() without an argument applies the accumulated batch centers."""
        center = Center(size=(1, 2), mode="mean", momentum=0.0)
        center.accumulate(torch.tensor([[1.0, 2.0]]))
        center.accumulate(torch.tensor([[3.0, 4.0]]))
        center.update()
        assert torch.all(center.value == torch.tensor([2.0, 3.0]))

    def test_update__nothing_accumulated(self) -> None:
        """update() is a no-op if nothing was accumulated."""
        center = Center(size=(1, 2), mode="mean", momentum=0.0)
        center.update(torch.tensor([[1.0, 2.0]]))
        center.update()
        assert torch.all(center.value == torch.tensor([1.0, 2.0]))

    @pytest.mark.parametrize("num_chunks", [1, 2, 4])
    def test_accumulate__equivalent_to_single_update(self, num_chunks: int) -> None:
        """Accumulating equal-sized chunks matches a single update on all of them."""
        torch.manual_seed(0)
        chunks = [torch.rand(8, 4) for _ in range(num_chunks)]

        accumulated = Center(size=(1, 4), momentum=0.9)
        for chunk in chunks:
            accumulated.accumulate(chunk)
        accumulated.apply_update()

        single = Center(size=(1, 4), momentum=0.9)
        single.update(torch.cat(chunks, dim=0))

        assert torch.allclose(accumulated.value, single.value)

    def test_apply_update__resets_accumulator(self) -> None:
        center = Center(size=(1, 2), mode="mean", momentum=0.9)
        center.accumulate(torch.tensor([[1.0, 2.0]]))
        center.apply_update()
        assert center._num_accumulated == 0
        assert center._batch_sum is None
        # A second apply_update must not change the center again.
        value = center.value.clone()
        center.apply_update()
        assert torch.all(center.value == value)

    def test_state_dict__accumulator_not_persisted(self) -> None:
        """The accumulator must stay out of the state dict for checkpoint compat."""
        center = Center(size=(1, 2), mode="mean")
        assert set(center.state_dict().keys()) == {"center"}

    @pytest.mark.DDP
    @pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available")
    def test_accumulate__unequal_elements_per_rank(self) -> None:
        # Ranks contribute different numbers of elements, as in IBOTPatchLoss where
        # the number of masked tokens varies.
        torch.manual_seed(0)
        chunks = [
            [torch.rand(num_elements, 4) for num_elements in counts]
            for counts in ([2, 10], [7, 1])
        ]
        args = [(rank, NUM_PROCESSES, chunks) for rank in range(NUM_PROCESSES)]

        accumulated = pytest.pool.starmap(_accumulate_worker, args)  # type: ignore[attr-defined]
        single = pytest.pool.starmap(_single_update_worker, args)  # type: ignore[attr-defined]

        assert all(torch.allclose(a, s) for a, s in zip(accumulated, single))
