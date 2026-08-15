"""Optimiser parameter groups."""

from __future__ import annotations

from typing import Any, Dict, List

from torch.nn import Module, Parameter

from lightly.models.utils import get_weight_decay_parameters

__all__ = ["param_groups"]


def param_groups(*modules: Module, weight_decay: float) -> List[Dict[str, Any]]:
    """Splits parameters into a decayed group and a group that is not decayed.

    Normalization parameters and biases are the ones left out, which is what
    every SSL reference implementation does and what the published numbers were
    produced with.

    Example:
        >>> optimizer = LARS(param_groups(backbone, head, weight_decay=1e-6), lr=4.8)

    Args:
        *modules: The modules whose parameters to group.
        weight_decay: The decay applied to everything except norms and biases.

    Returns:
        Two groups, ready to hand to an optimiser. Both carry an explicit
        ``weight_decay``, so the optimiser's own default never applies.
    """
    decayed: List[Parameter]
    not_decayed: List[Parameter]
    decayed, not_decayed = get_weight_decay_parameters(modules)
    return [
        {"name": "decay", "params": decayed, "weight_decay": weight_decay},
        {"name": "no_weight_decay", "params": not_decayed, "weight_decay": 0.0},
    ]
