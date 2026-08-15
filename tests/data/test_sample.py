from typing import Any, List, Tuple

import pytest
import torch

from lightly.data.sample import Sample, View, collate, legacy_collate


def item(target: int = 0, extras: bool = False) -> Tuple[List[View], int]:
    views = [
        View(torch.randn(3, 4, 4), extras={"grid": torch.zeros(2)} if extras else {}),
        View(torch.randn(3, 4, 4), extras={"grid": torch.ones(2)} if extras else {}),
    ]
    return views, target


def test_selectors_read_the_labels_a_transform_wrote() -> None:
    sample = Sample(
        views=[
            View(torch.randn(1), role="global"),
            View(torch.randn(1), role="local"),
            View(torch.randn(1), stream="text", role="local"),
        ]
    )
    assert len(sample.by_role("global")) == 1
    assert len(sample.by_role("local")) == 2
    assert len(sample.by_stream("text")) == 1
    assert sample.by_role("context") == []


def test_collate_stacks_data_and_keeps_the_view_order() -> None:
    sample = collate([item(target=i) for i in range(4)])
    assert [tuple(view.data.shape) for view in sample.views] == [
        (4, 3, 4, 4),
        (4, 3, 4, 4),
    ]
    assert torch.equal(sample.meta["target"], torch.tensor([0, 1, 2, 3]))


def test_collate_stacks_every_extras_entry() -> None:
    sample = collate([item(extras=True) for _ in range(4)])
    assert tuple(sample.views[1].extras["grid"].shape) == (4, 2)
    assert torch.equal(sample.views[1].extras["grid"], torch.ones(4, 2))


def test_collate_takes_views_without_a_target() -> None:
    sample = collate([[View(torch.randn(3, 4, 4))] for _ in range(2)])
    assert tuple(sample.views[0].data.shape) == (2, 3, 4, 4)
    assert sample.meta == {}


def test_collate_keeps_the_filename_of_a_three_tuple() -> None:
    batch = [(*item(target=i), f"{i}.jpg") for i in range(2)]
    sample = collate(batch)
    assert sample.meta["filename"] == ["0.jpg", "1.jpg"]


def test_a_ragged_view_count_is_refused() -> None:
    batch = [item(), ([View(torch.randn(3, 4, 4))], 0)]
    with pytest.raises(ValueError, match="different view counts"):
        collate(batch)


def test_a_view_labelled_differently_across_samples_is_refused() -> None:
    batch = [
        ([View(torch.randn(1), role="global")], 0),
        ([View(torch.randn(1), role="local")], 1),
    ]
    with pytest.raises(ValueError, match="view 0 is"):
        collate(batch)


def test_extras_that_appear_in_one_sample_only_are_refused() -> None:
    batch = [
        ([View(torch.randn(1), extras={"grid": torch.zeros(2)})], 0),
        ([View(torch.randn(1))], 1),
    ]
    with pytest.raises(ValueError, match="extras"):
        collate(batch)


def test_an_empty_batch_is_refused() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate([])


def test_legacy_collate_yields_the_one_x_tuple() -> None:
    views, labels, filenames = legacy_collate(
        [(*item(target=i), "a") for i in range(2)]
    )
    assert [tuple(view.shape) for view in views] == [(2, 3, 4, 4), (2, 3, 4, 4)]
    assert torch.equal(labels, torch.tensor([0, 1]))
    assert filenames == ["a", "a"]
