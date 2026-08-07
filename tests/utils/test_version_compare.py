import pytest

from lightly.utils import version_compare


class TestVersionCompare:
    def test_valid_versions(self) -> None:
        # general test of smaller than version numbers
        assert version_compare.version_compare("0.1.4", "1.2.0") == -1
        assert version_compare.version_compare("1.1.0", "1.2.0") == -1

        # test bigger than
        assert version_compare.version_compare("1.2.0", "1.1.0") == 1
        assert version_compare.version_compare("1.2.0", "0.1.4") == 1

        # test equal
        assert version_compare.version_compare("1.2.0", "1.2.0") == 0

    def test_invalid_versions(self) -> None:
        with pytest.raises(ValueError):
            version_compare.version_compare("1.2", "1.1.0")

        with pytest.raises(ValueError):
            version_compare.version_compare("1.2.0.1", "1.1.0")

        # test within same minor version and with special cases
        with pytest.raises(ValueError):
            assert version_compare.version_compare("1.0.7", "1.1.0.dev1") == -1

        with pytest.raises(ValueError):
            assert version_compare.version_compare("1.1.0.dev1", "1.1.0rc1") == -1
