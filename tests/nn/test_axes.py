from typing import List

import pytest
import torch
from torch import Tensor
from torch.nn import Module

from lightly.data.sample import View
from lightly.nn import encode


class Counter(Module):
    """Pools over space, and records the batch size of every call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: List[int] = []

    def forward(self, images: Tensor) -> Tensor:
        self.calls.append(images.size(0))
        return images.mean(dim=(-2, -1))


def test_same_shape_views_reach_the_module_in_one_call() -> None:
    module = Counter()
    views = [View(torch.randn(3, 4, 2, 2)) for _ in range(2)]
    features = encode(module, views, group_by="shape")
    assert module.calls == [6]
    assert [tuple(f.shape) for f in features] == [(3, 4), (3, 4)]


def test_different_shapes_are_grouped_separately() -> None:
    module = Counter()
    views = [View(torch.randn(3, 4, 2, 2)) for _ in range(2)]
    views += [View(torch.randn(2, 4, 1, 1))]
    encode(module, views, group_by="shape")
    assert module.calls == [6, 2]


def test_group_by_view_calls_once_per_view() -> None:
    module = Counter()
    views = [View(torch.randn(3, 3, 2, 2)) for _ in range(2)]
    encode(module, views, group_by="view")
    assert module.calls == [3, 3]


def test_features_come_back_in_view_order() -> None:
    module = Counter()
    small, large = View(torch.zeros(2, 4, 1, 1)), View(torch.ones(2, 4, 2, 2))
    features = encode(module, [small, large, small], group_by="shape")
    assert [f.shape[0] for f in features] == [2, 2, 2]
    assert torch.equal(features[0], features[2])


def test_embed_is_preferred_over_call() -> None:
    class WithEmbed(Counter):
        def embed(self, images: Tensor) -> Tensor:
            self.calls.append(-images.size(0))
            return images.mean(dim=(-2, -1))

    module = WithEmbed()
    encode(module, [View(torch.randn(3, 4, 2, 2))], group_by="shape")
    assert module.calls == [-3]


def test_unknown_grouping_raises() -> None:
    with pytest.raises(ValueError, match="group_by"):
        encode(Counter(), [View(torch.randn(1, 4, 2, 2))], group_by="fused")


def test_no_views_raises() -> None:
    with pytest.raises(ValueError, match="no views"):
        encode(Counter(), [])
