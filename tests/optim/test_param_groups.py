from typing import Set

import torch
from torch.nn import BatchNorm1d, Conv2d, LayerNorm, Linear, Module, Sequential

from lightly.optim import LARS, param_groups


class Model(Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = Conv2d(3, 4, kernel_size=3)
        self.norm = BatchNorm1d(4)
        self.head = Sequential(Linear(4, 4), LayerNorm(4))


def names_of(model: Module, wanted: Set[int]) -> Set[str]:
    return {name for name, p in model.named_parameters() if id(p) in wanted}


def test_norms_and_biases_are_not_decayed() -> None:
    model = Model()
    decay, no_decay = param_groups(model, weight_decay=1e-6)

    assert decay["weight_decay"] == 1e-6
    assert no_decay["weight_decay"] == 0.0
    assert names_of(model, {id(p) for p in decay["params"]}) == {
        "conv.weight",
        "head.0.weight",
    }
    assert names_of(model, {id(p) for p in no_decay["params"]}) == {
        "conv.bias",
        "norm.weight",
        "norm.bias",
        "head.0.bias",
        "head.1.weight",
        "head.1.bias",
    }


def test_every_parameter_lands_in_exactly_one_group() -> None:
    model = Model()
    groups = param_groups(model, weight_decay=1e-6)
    grouped = [id(p) for group in groups for p in group["params"]]
    assert sorted(grouped) == sorted(id(p) for p in model.parameters())


def test_several_modules_are_grouped_together() -> None:
    backbone, head = Model(), Linear(4, 2)
    decay, no_decay = param_groups(backbone, head, weight_decay=0.1)
    assert any(p is head.weight for p in decay["params"])
    assert any(p is head.bias for p in no_decay["params"])


def test_the_groups_are_what_an_optimiser_takes() -> None:
    model = Model()
    optimizer = LARS(param_groups(model, weight_decay=1e-6), lr=0.1, momentum=0.9)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [1e-6, 0.0]
    linear = model.head[0]
    assert isinstance(linear, Linear)
    linear.weight.grad = torch.ones_like(linear.weight)
    optimizer.step()
