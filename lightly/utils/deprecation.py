"""Helpers for emitting deprecation warnings."""

from __future__ import annotations

import warnings


def warn_deprecated(
    name: str, alternative: str, removed_in: str, stacklevel: int = 3
) -> None:
    """Warns that ``name`` is deprecated and will be removed.

    Uses ``FutureWarning`` rather than ``DeprecationWarning`` because the default
    warning filter shows ``FutureWarning`` to users, while it hides
    ``DeprecationWarning`` unless the warning originates from ``__main__``.

    Args:
        name:
            Name of the deprecated object.
        alternative:
            Name of the object to use instead.
        removed_in:
            The lightly version in which ``name`` will be removed.
        stacklevel:
            Stack level passed to :func:`warnings.warn`. The default of 3 points the
            warning at the caller when this helper is called directly from the
            deprecated object's ``__init__`` or from a module ``__getattr__``.
    """
    warnings.warn(
        f"{name} is deprecated and will be removed in lightly {removed_in}. "
        f"Use {alternative} instead.",
        FutureWarning,
        stacklevel=stacklevel,
    )
