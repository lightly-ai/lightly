import pytest

from lightly.utils.deprecation import warn_deprecated


def test_warn_deprecated() -> None:
    with pytest.warns(FutureWarning) as record:
        warn_deprecated(
            name="FeatureFoo",
            alternative="Use FeatureBar instead.",
            removed_in="1.6.0",
        )

    assert len(record) == 1
    assert (
        record[0].message.args[0]
        == "FeatureFoo is deprecated and will be removed in version 1.6.0. Use FeatureBar instead."
    )


def test_warn_deprecated__with_already_deprecated_in_name() -> None:
    with pytest.warns(FutureWarning) as record:
        warn_deprecated(
            name="FeatureFoo is deprecated",
            alternative="Use FeatureBar instead.",
            removed_in="1.6.0",
        )

    assert len(record) == 1
    assert (
        record[0].message.args[0]
        == "FeatureFoo is deprecated and will be removed in version 1.6.0. Use FeatureBar instead."
    )


def test_warn_deprecated__no_optional_args() -> None:
    with pytest.warns(FutureWarning) as record:
        warn_deprecated(name="FeatureFoo")

    assert len(record) == 1
    assert record[0].message.args[0] == "FeatureFoo is deprecated."
