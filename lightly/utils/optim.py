from typing import Any, Dict, Iterable, Optional

from torch.optim import Optimizer


def update_param_groups(
    optimizer: Optimizer,
    default_update: Optional[Dict[str, Any]] = None,
    updates: Optional[Iterable[Dict[str, Any]]] = None,
) -> None:
    """Update the values in the param groups of the optimizer.

    This is useful to update values following a schedule during training.

    Args:
        optimizer:
            A PyTorch optimizer.
        default_update:
            Key value pairs that will be updated for all param groups. The value is only
            updated if the respective key already exists in the param group.
        updates:
            A list of dicts with key value pairs. Each dict must contain a key "name"
            that specifies the name of the param group(s) that should be updated. All
            values of the dict will be updated in the param group(s) with the same name.
            If a key is in default_update and in updates, the value in updates has
            precedence.

    Examples:
        >>> optimizer = torch.optim.SGD(
        >>>     [
        >>>         {
        >>>             "name": "model",
        >>>             "params": params,
        >>>         },
        >>>         {
        >>>             "name": "model_no_weight_decay",
        >>>             "params": params_no_weight_decay,
        >>>             "weight_decay": 0.0,
        >>>         },
        >>>         {
        >>>             "name": "head",
        >>>             "params": head_params,
        >>>             "weight_decay": 0.5,
        >>>         },
        >>>     ],
        >>>     lr=0.1,
        >>>     weight_decay=0.2,
        >>> )
        >>>
        >>> update_param_groups(
        >>>     optimizer=optimizer,
        >>>     default_update={"weight_decay": 0.3},
        >>>     updates=[
        >>>         {"name": "model_no_weight_decay", "weight_decay": 0.0},
        >>>         {"name": "head", "weight_decay": 0.6},
        >>>     ]
        >>> )
        >>>
        >>> # Param group "model" has now weight decay 0.3
        >>> # Param group "model_no_weight_decay" has still weight decay 0.0
        >>> # Param group "head" has now weight decay 0.6

    """
    if default_update is None:
        default_update = {}
    if updates is None:
        updates = []

    # Update the optimizer's param_groups with the provided default values.
    for key, value in default_update.items():
        for param_group in optimizer.param_groups:
            # Update the value only if it already exists in the param_group, we don't
            # want to accidentally add new keys/values.
            if key in param_group:
                param_group[key] = value

    # Update the optimizer's param_groups with the provided updates.
    for update in updates:
        found_group = False
        name = update["name"]
        for param_group in optimizer.param_groups:
            if param_group.get("name") == name:
                found_group = True
                for key, value in update.items():
                    if key in param_group:
                        param_group[key] = value
                    else:
                        raise ValueError(
                            f"Key '{key}' not found in param group with name '{name}'."
                        )
        if not found_group:
            raise ValueError(f"No param group with name '{name}' in optimizer.")
