"""The gate: examples/simclr.py and benchmarks/simclr/ are one method.

The two files are written independently and restate each other on purpose. What
stops them drifting is not an import, it is this file. Every SimCLR in v1 failed
three of these:

    the two shipped heads differed by 4,922,112 parameters   load_state_dict raised
    split forward 2.582973, fused forward 2.757323           gradient cos sim 0.0837
    same seed, same image, 200 trials                        views differed on 108

What is compared and what is not
--------------------------------
Compared: the block classes, the view contract, the fused forward, the head's
input width against the backbone's output, and that the temperature each file
declares is the one its criterion carries.

Not compared: any value that belongs to the run rather than to the method.
Backbone family, image size, batch size, epochs, learning rate, the temperature's
value, the head's widths, blur, the normalization statistics and
``gather_distributed`` are all free to differ, and they do: the example is one
small configuration and the ``imagenet`` row is the paper's.

Where a number has to be shared for a comparison to mean anything, the row is
built out of the example's own constants. That asserts the two files compute the
same thing given the same numbers. It never asserts that they chose the same
numbers.

Not covered here: that each file excludes norms and biases from weight decay.
Both call ``lightly.optim.param_groups``, and ``tests/optim/test_param_groups.py``
is where that rule is checked.
"""

import random
from typing import Callable, List, NamedTuple

import pytest
import torch
from PIL import Image
from torch import Tensor
from torch.nn import Linear, Module
from torch.testing import assert_close

import examples.simclr as example
from benchmarks.simclr import benchmark
from benchmarks.simclr.datasets import SETTINGS, Setting
from lightly.backbones import TorchvisionResNetBackbone
from lightly.data.sample import Sample, View
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms import SimCLRTransform

SEED = 11


def setting_from_example() -> Setting:
    """The row the example would be, if the example were a row.

    Deliberately not an entry in ``SETTINGS``: it exists so the two files can be
    compared under one set of numbers, not so they share one.
    """
    return Setting(
        name="from_example",
        size=example.INPUT_SIZE,
        normalize=example.NORMALIZE,
        stem="small_image",
        blur=example.GAUSSIAN_BLUR,
        color_jitter_strength=example.COLOR_JITTER_STRENGTH,
        backbone="resnet18",
        feature_dim=example.FEATURE_DIM,
        hidden_dim=example.HIDDEN_DIM,
        out_dim=example.OUT_DIM,
        batch_size=example.BATCH_SIZE,
        epochs=example.EPOCHS,
        temperature=example.TEMPERATURE,
        weight_decay=example.WEIGHT_DECAY,
        optimizer="sgd",
        lr=example.LR,
        momentum=example.MOMENTUM,
        warmup_epochs=0,
        num_classes=10,
        knn_k=200,
        knn_t=0.1,
    )


def fixed_sample(seed: int, size: int, batch_size: int = 4) -> Sample:
    torch.manual_seed(seed)
    return Sample(
        views=[View(torch.randn(batch_size, 3, size, size)) for _ in range(2)],
        meta={"target": torch.zeros(batch_size, dtype=torch.long)},
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def synced(module: benchmark.SimCLR) -> benchmark.SimCLR:
    module.backbone.load_state_dict(example.backbone.state_dict())
    module.head.load_state_dict(example.head.state_dict())
    return module


# --------------------------------------------------------------------------- #
# One method, given one set of numbers.
# --------------------------------------------------------------------------- #


def test_weights_are_interchangeable() -> None:
    # v1: the two shipped heads differed by 4,922,112 parameters.
    module = benchmark.SimCLR(setting_from_example())
    module.backbone.load_state_dict(example.backbone.state_dict())
    module.head.load_state_dict(example.head.state_dict())


def test_same_loss_on_the_same_batch() -> None:
    # v1: split forward 2.582973, fused forward 2.757323.
    module = synced(benchmark.SimCLR(setting_from_example()))
    sample = fixed_sample(SEED, size=example.INPUT_SIZE)
    assert float(example.forward(sample).detach()) == pytest.approx(
        float(module(sample).detach()), abs=1e-6
    )


def test_same_views_from_the_same_seed() -> None:
    # v1: the same seed and the same image gave different views on 108 of 200.
    image = Image.new("RGB", (64, 64), color=(30, 90, 150))
    from_benchmark = benchmark.transform(setting_from_example())

    seed_everything(SEED)
    ours = example.transform(image)
    seed_everything(SEED)
    theirs = from_benchmark(image)

    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs):
        assert_close(a.data, b.data)


# --------------------------------------------------------------------------- #
# One method, each side reading its own numbers.
# --------------------------------------------------------------------------- #


class Side(NamedTuple):
    name: str
    backbone: TorchvisionResNetBackbone
    head: Module
    criterion: NTXentLoss
    transform: SimCLRTransform
    forward: Callable[[Sample], Tensor]
    declared_temperature: float
    size: int


def example_side() -> Side:
    return Side(
        name="example",
        backbone=example.backbone,
        head=example.head,
        criterion=example.criterion,
        transform=example.transform,
        forward=example.forward,
        declared_temperature=example.TEMPERATURE,
        size=example.INPUT_SIZE,
    )


def benchmark_side() -> Side:
    setting = SETTINGS["cifar10"]
    module = benchmark.SimCLR(setting)
    return Side(
        name=f"benchmark[{setting.name}]",
        backbone=module.backbone,
        head=module.head,
        criterion=module.criterion,
        transform=benchmark.transform(setting),
        forward=module.forward,
        declared_temperature=setting.temperature,
        size=setting.size,
    )


SIDES = [example_side, benchmark_side]
IDS = ["example", "benchmark"]


@pytest.mark.parametrize("build", SIDES, ids=IDS)
def test_the_same_blocks_are_assembled(build: Callable[[], Side]) -> None:
    side = build()
    assert isinstance(side.head, SimCLRProjectionHead)
    assert isinstance(side.criterion, NTXentLoss)
    assert isinstance(side.transform, SimCLRTransform)
    assert hasattr(side.backbone, "embed")


@pytest.mark.parametrize("build", SIDES, ids=IDS)
def test_two_views_with_the_default_role(build: Callable[[], Side]) -> None:
    side = build()
    views = side.transform(Image.new("RGB", (64, 64)))
    assert len(views) == 2
    assert [view.role for view in views] == ["view", "view"]
    assert [view.stream for view in views] == ["image", "image"]
    assert views[0].data.shape == views[1].data.shape


@pytest.mark.parametrize("build", SIDES, ids=IDS)
def test_the_head_reads_the_backbone_width(build: Callable[[], Side]) -> None:
    side = build()
    images = torch.randn(2, 3, side.size, side.size)
    assert side.backbone.embed(images).shape[1] == side.backbone.feature_dim
    first_linear = next(
        module for module in side.head.modules() if isinstance(module, Linear)
    )
    assert first_linear.in_features == side.backbone.feature_dim


@pytest.mark.parametrize("build", SIDES, ids=IDS)
def test_both_views_reach_the_encoder_in_one_call(build: Callable[[], Side]) -> None:
    # BatchNorm has to see 2N. One call per view gives it N, and the gradients of
    # the two agree at cosine similarity 0.0837. This drives each side's own
    # forward, not encode: a test that calls encode itself proves nothing about
    # the file it is meant to be checking.
    side = build()
    calls: List[int] = []
    handle = side.backbone.register_forward_pre_hook(
        lambda _module, inputs: calls.append(inputs[0].size(0))
    )
    try:
        side.forward(fixed_sample(SEED, size=side.size, batch_size=3))
    finally:
        handle.remove()
    assert calls == [6]


@pytest.mark.parametrize("build", SIDES, ids=IDS)
def test_the_temperature_is_stated_not_defaulted(build: Callable[[], Side]) -> None:
    # v1: both examples took NTXentLoss()'s default 0.5 while the benchmark
    # passed 0.1, and nothing noticed.
    side = build()
    assert side.criterion.temperature == side.declared_temperature


def test_every_row_states_a_width_its_head_can_read() -> None:
    for name, setting in SETTINGS.items():
        module = benchmark.SimCLR(setting)
        assert module.backbone.feature_dim == setting.feature_dim, name
        first_linear = next(
            child for child in module.head.modules() if isinstance(child, Linear)
        )
        assert first_linear.in_features == setting.feature_dim, name
