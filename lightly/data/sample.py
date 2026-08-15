"""The batch contract: a sample is a list of typed views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor

__all__ = ["Sample", "View", "collate", "legacy_collate"]


@dataclass
class View:
    """One view of one sample, labelled with what it is.

    A single item holds ``data`` at ``(C, H, W)``. After collation the same field
    holds ``(B, C, H, W)``.

    Attributes:
        data:
            The view itself.
        stream:
            The modality the view came from: ``image``, ``text``, ``audio``,
            ``state`` or ``action``.
        role:
            What the method does with the view: ``view``, ``global``, ``local``,
            ``context``, ``target`` or ``anchor``.
        extras:
            Whatever the transform emitted alongside the view, such as a mask, a
            grid or patch ids. Collation stacks every entry.
    """

    data: Tensor
    stream: str = "image"
    role: str = "view"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sample:
    """A batch of views, plus what belongs to the sample rather than to a view.

    Attributes:
        views:
            The views, in the order the transform produced them.
        meta:
            Per-sample values such as the target, the filename or an episode id.
    """

    views: list[View]
    meta: dict[str, Any] = field(default_factory=dict)

    def by_role(self, role: str) -> list[View]:
        """Returns the views with the given role, in view order.

        Args:
            role: The role to select.

        Returns:
            The matching views, empty if there are none.
        """
        return [view for view in self.views if view.role == role]

    def by_stream(self, stream: str) -> list[View]:
        """Returns the views with the given stream, in view order.

        Args:
            stream: The stream to select.

        Returns:
            The matching views, empty if there are none.
        """
        return [view for view in self.views if view.stream == stream]


def _stack(values: Sequence[Any]) -> Any:
    """Stacks tensors, leaves anything else as a list."""
    if all(isinstance(value, Tensor) for value in values):
        return torch.stack(list(values))
    return list(values)


def _merge(views: Sequence[View], position: int) -> View:
    """Merges the view at one position across the samples of a batch.

    The collate only sees sample ``0`` as a declaration, so a mismatch is
    reported against it rather than against a contract.

    Args:
        views: The view at this position, one per sample.
        position: The position, used in the error message.

    Returns:
        One view holding the stacked data and extras.

    Raises:
        ValueError: If the views disagree on stream, role or extras.
    """
    first = views[0]
    for index, view in enumerate(views[1:], start=1):
        if (view.stream, view.role) != (first.stream, first.role):
            raise ValueError(
                f"view {position} is ({first.stream!r}, {first.role!r}) in sample 0 "
                f"and ({view.stream!r}, {view.role!r}) in sample {index}"
            )
        if set(view.extras) != set(first.extras):
            raise ValueError(
                f"view {position} has extras {sorted(first.extras)} in sample 0 "
                f"and {sorted(view.extras)} in sample {index}"
            )
    return View(
        data=torch.stack([view.data for view in views]),
        stream=first.stream,
        role=first.role,
        extras={
            key: _stack([view.extras[key] for view in views]) for key in first.extras
        },
    )


def _views_and_meta(item: Any) -> tuple[Sequence[View], dict[str, Any]]:
    """Splits one dataset item into its views and its per-sample values."""
    if isinstance(item, View):
        return [item], {}
    if isinstance(item, (list, tuple)) and all(isinstance(x, View) for x in item):
        return item, {}
    views, rest = item[0], item[1:]
    meta: dict[str, Any] = {}
    if len(rest) > 0:
        meta["target"] = rest[0]
    if len(rest) > 1:
        meta["filename"] = rest[1]
    return views, meta


def collate(batch: Sequence[Any]) -> Sample:
    """Collates dataset items whose transform returns views.

    Takes no configuration: everything the batch needs to be assembled arrived
    with the data. Views are matched by position, and both ``data`` and every
    ``extras`` entry are stacked.

    Args:
        batch:
            The items, each one ``list[View]`` or a tuple whose first element is
            ``list[View]``. A second element becomes ``meta["target"]`` and a
            third becomes ``meta["filename"]``.

    Returns:
        One sample holding the batched views.

    Raises:
        ValueError: If the batch is empty or the items disagree on view count.
    """
    if len(batch) == 0:
        raise ValueError("collate received an empty batch")

    split = [_views_and_meta(item) for item in batch]
    views_per_sample = [views for views, _ in split]

    counts = {len(views) for views in views_per_sample}
    if len(counts) > 1:
        raise ValueError(f"samples in the batch have different view counts: {counts}")

    sample = Sample(
        views=[
            _merge(views, position)
            for position, views in enumerate(zip(*views_per_sample))
        ]
    )
    for key in split[0][1]:
        values = [meta[key] for _, meta in split]
        sample.meta[key] = (
            _stack(values)
            if key != "target"
            else (
                torch.stack(values)
                if isinstance(values[0], Tensor)
                else torch.as_tensor(values)
            )
        )
    return sample


def legacy_collate(batch: Sequence[Any]) -> tuple[list[Tensor], Tensor, list[str]]:
    """Collates into the 1.x ``(views, labels, filenames)`` tuple.

    A shim for training loops written against the old batch type, kept for the
    whole 2.x line.

    Args:
        batch: The items, as for :func:`collate`.

    Returns:
        The views as bare tensors, the labels and the filenames.
    """
    sample = collate(batch)
    labels = sample.meta.get("target", torch.empty(0, dtype=torch.long))
    filenames = sample.meta.get("filename", [])
    return [view.data for view in sample.views], labels, filenames
