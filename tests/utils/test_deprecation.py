import pytest

from lightly.utils.deprecation import warn_deprecated


def test_warn_deprecated() -> None:
    with pytest.warns(FutureWarning) as record:
        warn_deprecated("Old", "New", removed_in="1.7.0")
    assert len(record) == 1
    message = str(record[0].message)
    assert "Old is deprecated" in message
    assert "removed in lightly 1.7.0" in message
    assert "Use New instead" in message
