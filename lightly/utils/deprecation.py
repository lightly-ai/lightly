import warnings


def warn_deprecated(
    name: str,
    alternative: str = "",
    removed_in: str = "",
    stacklevel: int = 3,
) -> None:
    """Warns that a feature is deprecated using a FutureWarning.

    Args:
        name:
            Name or description of the deprecated feature (e.g., "BYOL").
        alternative:
            Suggested alternative to use instead.
        removed_in:
            Version when the feature will be removed.
        stacklevel:
            Stack level passed to warnings.warn.
    """
    if "deprecated" in name.lower():
        msg = name
    elif name.endswith(" is") or name.endswith(" are"):
        msg = f"{name} deprecated"
    else:
        msg = f"{name} is deprecated"

    if removed_in:
        msg += f" and will be removed in version {removed_in}"
    msg += "."

    if alternative:
        msg += f" {alternative}"

    warnings.warn(msg, category=FutureWarning, stacklevel=stacklevel)
