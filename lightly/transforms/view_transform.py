"""The base class for transforms that return labelled views."""

from __future__ import annotations

from typing import Any, Callable, List, Sequence

from lightly.data.sample import View

__all__ = ["ViewTransform"]


class ViewTransform:
    """Transforms an input into several views, each labelled with what it is.

    A view that is built is a view that is named, on the same line. Symmetric
    methods are finished at that: every view takes the default role. Asymmetric
    ones override ``__call__`` and state the asymmetry they already know, so no
    training loop has to recover it from ``views[:2]``.

    ``__call__`` takes ``*args`` so a transform that consumes an image and a mask
    together needs no second base class.

    Args:
        transforms:
            One callable per view. Each is called with whatever ``__call__``
            receives.
    """

    def __init__(self, transforms: Sequence[Callable[..., Any]]) -> None:
        self.transforms = transforms

    def __call__(self, *args: Any) -> List[View]:
        """Transforms the input into one view per transform.

        Args:
            *args: The input, passed on to every transform unchanged.

        Returns:
            One view per transform, in order.
        """
        return [View(transform(*args)) for transform in self.transforms]
