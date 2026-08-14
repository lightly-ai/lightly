from lightly.utils.deprecation import warn_deprecated


def raise_active_learning_deprecation_warning() -> None:
    warn_deprecated(
        "Active learning via the lightly package",
        "the Lightly Solution",
        removed_in="1.7.0",
    )
