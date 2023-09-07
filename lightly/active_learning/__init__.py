import warnings


def raise_active_learning_deprecation_warning():
    warnings.warn(
        "Using active learning via the lightly package is deprecated and will be removed soon. "
        "Please use the Lightly Solution instead. "
        "See https://docs.lightly.ai for more information and tutorials on doing active learning.",
        DeprecationWarning,
        stacklevel=2,
    )
