from typing import Tuple

import pytest
from torch.nn import Linear
from torch.optim import SGD

from lightly.schedulers import CosineWarmupScheduler, scheduler_utils


def test_init_schedulers__validate_schedulers() -> None:
    _, _, optimizer = _get_schedulers_and_optimizer()
    # This should not raise an error.
    scheduler_utils.init_schedulers(optimizer)


def test_init_schedulers__validate_schedulers__error() -> None:
    scheduler = CosineWarmupScheduler(max_steps=10)
    optimizers = SGD(
        [
            {
                "name": "param",
                "params": Linear(1, 1).parameters(),
                "weigth_decay": scheduler,  # Typo, scheduler must end with "_scheduler"
            }
        ],
        lr=1.0,
    )
    with pytest.raises(ValueError):
        scheduler_utils.init_schedulers(optimizers)


def test_init_schedulers__state_dict() -> None:
    scheduler1, scheduler2, optimizer = _get_schedulers_and_optimizer()
    scheduler_utils.init_schedulers(optimizer)

    # Assert that the state dict contains the state dicts of the schedulers and not the
    # schedulers themselves.
    state_dict = optimizer.state_dict()
    assert (
        state_dict["param_groups"][1]["weight_decay_scheduler"]
        == scheduler1.state_dict()
    )
    assert (
        state_dict["param_groups"][2]["weight_decay_scheduler"]
        == scheduler2.state_dict()
    )


def test_init__schedulers__load_state_dict() -> None:
    scheduler1, scheduler2, optimizer = _get_schedulers_and_optimizer()
    scheduler_utils.init_schedulers(optimizer)

    # Modify the state dict to check if the schedulers are updated correctly.
    scheduler_1_state_dict = scheduler1.state_dict()
    scheduler_1_state_dict["max_steps"] = 30
    scheduler_2_state_dict = scheduler2.state_dict()
    scheduler_2_state_dict["max_steps"] = 40
    state_dict = optimizer.state_dict()
    state_dict["param_groups"][1]["weight_decay_scheduler"] = scheduler_1_state_dict
    state_dict["param_groups"][2]["weight_decay_scheduler"] = scheduler_2_state_dict
    optimizer.load_state_dict(state_dict)

    # Assert that the state dicts have been loaded correctly.
    assert scheduler1.state_dict() == scheduler_1_state_dict
    assert scheduler2.state_dict() == scheduler_2_state_dict
    # Assert that the same scheduler instances remain in the optimizer param groups.
    assert optimizer.param_groups[1]["weight_decay_scheduler"] == scheduler1
    assert optimizer.param_groups[2]["weight_decay_scheduler"] == scheduler2


def test_init_schedulers__step() -> None:
    scheduler1, scheduler2, optimizer = _get_schedulers_and_optimizer()
    scheduler_utils.init_schedulers(optimizer)

    # Assert that the initial values have been set correctly.
    assert scheduler1.current_step == 1
    assert scheduler2.current_step == 1
    param_groups = optimizer.param_groups
    assert param_groups[1]["weight_decay"] == scheduler1.get_value(step=1)
    assert param_groups[2]["weight_decay"] == scheduler2.get_value(step=1)

    # Assert that the schedulers stepped.
    optimizer.step()
    assert scheduler1.current_step == 2
    assert scheduler2.current_step == 2
    assert param_groups[1]["weight_decay"] == scheduler1.get_value(step=2)
    assert param_groups[2]["weight_decay"] == scheduler2.get_value(step=2)


def _get_schedulers_and_optimizer() -> (
    Tuple[CosineWarmupScheduler, CosineWarmupScheduler, SGD]
):
    scheduler1 = CosineWarmupScheduler(max_steps=10)
    scheduler2 = CosineWarmupScheduler(max_steps=20)
    optimizer = SGD(
        [
            {
                "name": "param1",
                "params": Linear(1, 1).parameters(),
            },
            {
                "name": "param2",
                "params": Linear(1, 1).parameters(),
                "weight_decay_scheduler": scheduler1,
            },
            {
                "name": "param3",
                "params": Linear(1, 1).parameters(),
                "weight_decay_scheduler": scheduler2,
            },
        ],
        lr=1.0,
    )
    return scheduler1, scheduler2, optimizer
