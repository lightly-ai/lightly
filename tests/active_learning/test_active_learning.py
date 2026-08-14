import pytest

from lightly.active_learning import raise_active_learning_deprecation_warning


def test_raise_active_learning_deprecation_warning() -> None:
    with pytest.warns(FutureWarning, match="deprecated"):
        raise_active_learning_deprecation_warning()
