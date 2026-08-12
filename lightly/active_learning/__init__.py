from lightly.utils.deprecation import warn_deprecated


def raise_active_learning_deprecation_warning() -> None:
    warn_deprecated(
        name="Using active learning via the lightly package is",
        alternative=(
            "Please use the Lightly Solution instead. See https://docs.lightly.ai "
            "for more information and tutorials on doing active learning."
        ),
        removed_in="1.6.0",
        stacklevel=3,
    )
