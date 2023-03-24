import warnings

def raise_active_learning_deprecation_warning():
    print("asdf")
    warnings.warn(
        "Using active learning via the lightly package is deprecated and will be removed soon "
        "Please use the Lightly Solution instead. "
        "See https://docs.lightly.ai for more information including tutorials on doing active learning.",
        DeprecationWarning,
        stacklevel=2,
    )

raise_active_learning_deprecation_warning()