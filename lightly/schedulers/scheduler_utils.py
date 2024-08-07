from typing import Any, Dict

from torch.optim import Optimizer

from lightly.schedulers.scheduler import Scheduler

_SCHEDULER_SUFFIX = "_scheduler"


def init_schedulers(optimizer: Optimizer) -> None:
    _validate_schedulers(optimizer=optimizer)
    _update_scheduler_values(optimizer=optimizer, do_step=False)
    if _has_schedulers(optimizer=optimizer):
        # TODO(Guarin, 08/24): Newer versions of PyTorch have hooks to customize
        # state dict and step behavior with:
        # - optimizer.register_state_dict_post_hook (since 2.1.0)
        # - optimizer.register_load_state_dict_pre_hook (since 2.1.0)
        # - optimizer.register_step_post_hook (since 2.0.0)
        #
        # We use wrappers instead to also support older versions. In the future we
        # should check if the hooks are available and use them instead.
        optimizer.state_dict = _state_dict_wrapper(state_dict_fn=optimizer.state_dict)
        optimizer.load_state_dict = _load_state_dict_wrapper(
            optimizer=optimizer, load_state_dict_fn=optimizer.load_state_dict
        )
        optimizer.step = _step_wrapper(optimizer=optimizer, step_fn=optimizer.step)


def _validate_schedulers(optimizer: Optimizer) -> None:
    """Check that the schedulers in the optimizer's param_groups have the correct name."""
    for group in optimizer.param_groups:
        for scheduler_name, scheduler in group.items():
            if isinstance(scheduler, Scheduler) and not scheduler_name.endswith(
                _SCHEDULER_SUFFIX
            ):
                raise ValueError(
                    f"Scheduler name '{scheduler_name}' must end with "
                    f"'{_SCHEDULER_SUFFIX}'."
                )


def _update_scheduler_values(optimizer: Optimizer, do_step: bool) -> None:
    """Set the values of the schedulers in the optimizer's param_groups.

    Args:
        optimizer: The optimizer whose param_groups contain the schedulers.
        do_step: Whether to call the step method of the schedulers.
    """
    for group in optimizer.param_groups:
        for scheduler_name in group:
            scheduler = group[scheduler_name]
            if isinstance(scheduler, Scheduler):
                if do_step:
                    scheduler.step()
                value_name = scheduler_name.rstrip(_SCHEDULER_SUFFIX)
                group[value_name] = scheduler.get_value(step=scheduler.current_step)


def _has_schedulers(optimizer: Optimizer) -> bool:
    """Returns True if any of the optimizer's param_groups contain a scheduler."""
    return any(
        isinstance(scheduler, Scheduler)
        for group in optimizer.param_groups
        for scheduler in group.values()
    )


def _state_dict_wrapper(state_dict_fn):
    """Wraps the state_dict method of an optimizer to save the state of schedulers."""

    def new_state_dict_fn():
        state_dict = state_dict_fn()

        # Save state_dict of schedulers.
        for group in state_dict["param_groups"]:
            schedulers = {
                scheduler_name: scheduler
                for scheduler_name, scheduler in group.items()
                if isinstance(scheduler, Scheduler)
            }
            for scheduler_name, scheduler in schedulers.items():
                group[scheduler_name] = scheduler.state_dict()

        return state_dict

    return new_state_dict_fn


def _load_state_dict_wrapper(optimizer: Optimizer, load_state_dict_fn):
    """Wraps the load_state_dict method of an optimizer to load the state of schedulers."""

    def new_load_state_dict_fn(state_dict: Dict[str, Any]) -> None:
        # Save the original param_groups as they contain the scheduler instances.
        param_groups = optimizer.param_groups.copy()
        # Call original optimizer.load_state_dict. This overwrites scheduler instances
        # with scheduler state dicts in optimizer.param_groups.
        load_state_dict_fn(state_dict)
        # Replace scheduler state dicts with scheduler instances.
        for group, group_state_dict in zip(param_groups, optimizer.param_groups):
            assert group.keys() == group_state_dict.keys()
            for scheduler_name, scheduler in group.items():
                if isinstance(scheduler, Scheduler):
                    scheduler.load_state_dict(group_state_dict[scheduler_name])
                    # Replace dict with instance.
                    group_state_dict[scheduler_name] = scheduler

    return new_load_state_dict_fn


def _step_wrapper(optimizer: Optimizer, step_fn):
    """Wraps the step method of an optimizer to update the schedulers and param_groups
    values after each optimizer step.
    """

    def new_step_fn(*args, **kargs) -> None:
        result = step_fn(*args, **kargs)
        _update_scheduler_values(optimizer=optimizer, do_step=True)
        return result

    return new_step_fn
