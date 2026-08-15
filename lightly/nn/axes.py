"""Encoding a list of views without hand-rolling the batch axis."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import Tensor

from lightly.data.sample import View

__all__ = ["encode"]

_GROUPINGS = ("shape", "view")


def _call(module: Any, images: Tensor) -> Tensor:
    """Calls ``embed`` when the module has one, otherwise the module itself."""
    embed = getattr(module, "embed", None)
    features: Tensor = embed(images) if callable(embed) else module(images)
    return features


def encode(
    module: Any, views: Sequence[View], *, group_by: str = "shape"
) -> List[Tensor]:
    """Encodes views, calling the module once per group rather than once per view.

    The grouping is what a boolean cannot express. Concatenating every view fails
    the moment two views have different sizes, and one call per view gives
    BatchNorm ``N`` samples where the published number was produced with ``2N``:

        >>> encode(backbone, sample.views, group_by="shape")  # SimCLR: one call at 2N
        >>> encode(student, globals_ + locals_, group_by="shape")  # DINO: two calls

    Args:
        module:
            What to call. ``module.embed`` is used when it exists, so a
            ``Backbone`` and a plain ``nn.Module`` both work.
        views:
            The views to encode. Their ``data`` is batched, ``(B, C, H, W)``.
        group_by:
            ``shape`` puts views of equal shape into one call. ``view`` calls the
            module once per view.

    Returns:
        One tensor per view, in the order the views were given.

    Raises:
        ValueError: If ``views`` is empty or ``group_by`` is not a known grouping.
    """
    if len(views) == 0:
        raise ValueError("encode received no views")
    if group_by not in _GROUPINGS:
        raise ValueError(f"group_by must be one of {_GROUPINGS}, got {group_by!r}")

    if group_by == "view":
        groups = [[index] for index in range(len(views))]
    else:
        by_shape: Dict[Tuple[int, ...], List[int]] = {}
        for index, view in enumerate(views):
            by_shape.setdefault(tuple(view.data.shape[1:]), []).append(index)
        groups = list(by_shape.values())

    features: List[Tensor] = [torch.empty(0)] * len(views)
    for indices in groups:
        sizes = [views[index].data.size(0) for index in indices]
        batched = _call(module, torch.cat([views[index].data for index in indices]))
        for index, chunk in zip(indices, batched.split(sizes)):
            features[index] = chunk
    return features
