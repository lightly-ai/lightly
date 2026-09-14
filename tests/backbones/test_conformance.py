"""The conformance suite every backbone adapter is registered in.

mypy checks that a member exists with the right signature. It cannot check that
the member does what the contract says: a ``raise NotImplementedError`` is a
body, and two adapters can return different ranks from the same name. That is
what this suite is for.

Adding a family is an adapter and one line in ``ADAPTERS``.
"""

from typing import Callable, List, NamedTuple, Tuple

import pytest
import torch
from torch import Tensor
from torchvision.models import resnet18

from lightly.backbones import (
    Backbone,
    DenseBackbone,
    TorchvisionResNetBackbone,
    small_image_stem,
)


class Adapter(NamedTuple):
    name: str
    # Widen this to a union as more families are adapted.
    build: Callable[[], TorchvisionResNetBackbone]
    input_size: int
    dense: bool


ADAPTERS: List[Adapter] = [
    Adapter(
        name="torchvision_resnet",
        build=lambda: TorchvisionResNetBackbone(resnet18()),
        input_size=64,
        dense=True,
    ),
    Adapter(
        name="torchvision_resnet_small_stem",
        build=lambda: TorchvisionResNetBackbone(small_image_stem(resnet18())),
        input_size=32,
        dense=True,
    ),
]

IDS = [adapter.name for adapter in ADAPTERS]
DENSE_ADAPTERS = [adapter for adapter in ADAPTERS if adapter.dense]
DENSE_IDS = [adapter.name for adapter in DENSE_ADAPTERS]


@pytest.mark.parametrize("adapter", ADAPTERS, ids=IDS)
def test_satisfies_backbone(adapter: Adapter) -> None:
    assert isinstance(adapter.build(), Backbone)


@pytest.mark.parametrize("adapter", ADAPTERS, ids=IDS)
def test_embed_returns_feature_dim_columns(adapter: Adapter) -> None:
    backbone = adapter.build()
    images = torch.randn(2, 3, adapter.input_size, adapter.input_size)
    features = backbone.embed(images)
    assert features.shape == (2, backbone.feature_dim)


@pytest.mark.parametrize("adapter", ADAPTERS, ids=IDS)
def test_embed_is_deterministic_in_eval(adapter: Adapter) -> None:
    backbone = adapter.build().eval()
    images = torch.randn(2, 3, adapter.input_size, adapter.input_size)
    with torch.no_grad():
        assert torch.equal(backbone.embed(images), backbone.embed(images))


@pytest.mark.parametrize("adapter", ADAPTERS, ids=IDS)
def test_calling_the_module_equals_embed(adapter: Adapter) -> None:
    backbone = adapter.build().eval()
    images = torch.randn(2, 3, adapter.input_size, adapter.input_size)
    with torch.no_grad():
        assert torch.equal(backbone(images), backbone.embed(images))


@pytest.mark.parametrize("adapter", DENSE_ADAPTERS, ids=DENSE_IDS)
def test_satisfies_dense_backbone(adapter: Adapter) -> None:
    assert isinstance(adapter.build(), DenseBackbone)


@pytest.mark.parametrize("adapter", DENSE_ADAPTERS, ids=DENSE_IDS)
def test_feature_map_flattens_the_grid_it_returns(adapter: Adapter) -> None:
    backbone = adapter.build()
    images = torch.randn(2, 3, adapter.input_size, adapter.input_size)
    features, (height, width) = backbone.feature_map(images)
    assert features.shape == (2, height * width, backbone.feature_dim)


@pytest.mark.parametrize("adapter", ADAPTERS, ids=IDS)
def test_no_member_raises_not_implemented(adapter: Adapter) -> None:
    backbone = adapter.build()
    images = torch.randn(2, 3, adapter.input_size, adapter.input_size)
    members: List[Tuple[str, Callable[[Tensor], object]]] = [("embed", backbone.embed)]
    if adapter.dense:
        members.append(("feature_map", backbone.feature_map))
    for name, member in members:
        try:
            member(images)
        except NotImplementedError:
            pytest.fail(f"{adapter.name}.{name} raises NotImplementedError")
