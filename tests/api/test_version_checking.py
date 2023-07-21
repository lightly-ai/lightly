import os
import time

import pytest
from pytest_mock import MockerFixture
from urllib3.exceptions import MaxRetryError

from lightly.api import _version_checking
from lightly.openapi_generated.swagger_client.api import VersioningApi


# Overwrite the mock_versioning_api fixture from conftest.py that is applied by default
# for all tests as we want to test the functionality of the versioning api.
@pytest.fixture(autouse=True)
def mock_versioning_api():
    return


@pytest.mark.disable_mock_versioning_api
def test_is_latest_version(mocker: MockerFixture) -> None:
    mocker.patch.object(
        _version_checking.VersioningApi, "get_latest_pip_version", return_value="1.2.8"
    )
    assert _version_checking.is_latest_version("1.2.8")
    assert not _version_checking.is_latest_version("1.2.7")
    assert not _version_checking.is_latest_version("1.1.8")
    assert not _version_checking.is_latest_version("0.2.8")


def test_is_compatible_version(mocker: MockerFixture) -> None:
    mocker.patch.object(
        _version_checking.VersioningApi,
        "get_minimum_compatible_pip_version",
        return_value="1.2.8",
    )
    assert _version_checking.is_compatible_version("1.2.8")
    assert not _version_checking.is_compatible_version("1.2.7")
    assert not _version_checking.is_compatible_version("1.1.8")
    assert not _version_checking.is_compatible_version("0.2.8")


def test_get_latest_version(mocker: MockerFixture) -> None:
    mocker.patch.object(
        _version_checking.VersioningApi, "get_latest_pip_version", return_value="1.2.8"
    )
    assert _version_checking.get_latest_version("1.2.8") == "1.2.8"


def test_get_latest_version__timeout(mocker: MockerFixture) -> None:
    mocker.patch.dict(os.environ, {"LIGHTLY_SERVER_LOCATION": "invalid-url"})
    start = time.perf_counter()
    with pytest.raises(MaxRetryError):
        # Urllib3 raises a timeout error (connection refused) for invalid URLs.
        _version_checking.get_latest_version("1.2.8", timeout_sec=0.1)
    end = time.perf_counter()
    assert end - start < 0.2  # give some slack for timeout


def test_get_minimum_compatible_version(mocker: MockerFixture) -> None:
    mocker.patch.object(
        _version_checking.VersioningApi,
        "get_minimum_compatible_pip_version",
        return_value="1.2.8",
    )

    assert _version_checking.get_minimum_compatible_version() == "1.2.8"


def test_get_minimum_compatible_version__timeout(mocker: MockerFixture) -> None:
    mocker.patch.dict(os.environ, {"LIGHTLY_SERVER_LOCATION": "invalid-url"})
    start = time.perf_counter()
    with pytest.raises(MaxRetryError):
        # Urllib3 raises a timeout error (connection refused) for invalid URLs.
        _version_checking.get_minimum_compatible_version(timeout_sec=0.1)
    end = time.perf_counter()
    assert end - start < 0.2  # give some slack for timeout


def test_check_is_latest_version_in_background(mocker: MockerFixture) -> None:
    spy_is_latest_version = mocker.spy(_version_checking, "is_latest_version")
    _version_checking.check_is_latest_version_in_background("1.2.8")
    spy_is_latest_version.assert_called_once_with(current_version="1.2.8")


def test__get_versioning_api() -> None:
    assert isinstance(_version_checking._get_versioning_api(), VersioningApi)
