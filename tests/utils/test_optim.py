from typing import Any, Dict, List

import pytest
from torch.nn import Linear
from torch.optim import SGD

from lightly.utils import optim


def test_update_param_groups() -> None:
    params: List[Dict[Any, Any]] = [
        {
            "name": "default",
            "params": Linear(1, 1).parameters(),
        },
        {
            "name": "no_wd",
            "params": Linear(1, 1).parameters(),
            "weight_decay": 0.0,
        },
        {
            "name": "wd",
            "params": Linear(1, 1).parameters(),
            "weight_decay": 0.5,
        },
    ]
    optimizer = SGD(
        params=params,
        lr=0.1,
        weight_decay=0.2,
    )

    assert optimizer.param_groups[0]["weight_decay"] == 0.2
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    assert optimizer.param_groups[2]["weight_decay"] == 0.5

    optim.update_param_groups(
        optimizer=optimizer,
        default_update={"weight_decay": 0.3},
        updates=[
            {"name": "no_wd", "weight_decay": 0.0},
            {"name": "wd", "weight_decay": 0.6},
        ],
    )

    assert optimizer.param_groups[0]["weight_decay"] == 0.3
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    assert optimizer.param_groups[2]["weight_decay"] == 0.6


def test_update_param_groups__default_no_add_entry() -> None:
    """Test that update_param_groups does not add new entries to param groups."""
    optimizer = SGD([{"name": "model", "params": Linear(1, 1).parameters()}], lr=0.1)
    optim.update_param_groups(
        optimizer=optimizer,
        default_update={"unknown": 1.0},
    )
    assert "unknown" not in optimizer.param_groups[0]


@pytest.mark.parametrize(  # type: ignore[misc]
    "updates, match",
    [
        ([{"name": "unknown"}], "No param group with name 'unknown' in optimizer."),
        (
            [{"name": "model", "unknown": 1.0}],
            "Key 'unknown' not found in param group with name 'model'.",
        ),
    ],
)
def test_update_param_groups__error(updates: List[Dict[str, Any]], match: str) -> None:
    optimizer = SGD([{"name": "model", "params": Linear(1, 1).parameters()}], lr=0.1)
    with pytest.raises(ValueError, match=match):
        optim.update_param_groups(optimizer=optimizer, updates=updates)
