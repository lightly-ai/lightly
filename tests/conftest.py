# content of conftest.py

import os
from unittest import mock

import pytest
import torch.multiprocessing as mp

# Distributed (DDP) test pool, see #1982. The flag and gloo setup live in
# tests/ddp_helpers.py so they can be typed and shared with the test modules;
# the session hooks below only start/stop the pool.
from tests.ddp_helpers import NUM_PROCESSES, USE_PYTEST_POOL, setup_ddp, teardown_ddp


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    """Pytest configuration hook, for docs see:
    https://docs.pytest.org/en/7.1.x/reference/reference.html#pytest.hookspec.pytest_configure

    This hook runs before any tests are collected or run.
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "DDP: mark test to run on the shared gloo process pool"
    )

    # This avoids running a version check when importing anything from lightly.
    # See lightly/__init__.py. Note that we cannot mock the version check
    # in __init__.py because it already runs when pytest collects the tests. This
    # happens before any fixtures are applied and therefore the mocking is not yet in
    # place.
    os.environ["LIGHTLY_DID_VERSION_CHECK"] = "True"

    # This avoids sending requests to the API.
    os.environ["LIGHTLY_SERVER_LOCATION"] = "https://dummy-url"


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="module", autouse=True)
def mock_versioning_api():
    """Fixture that is applied to all tests and mocks the versioning API.

    This is necessary because every time an ApiWorkflowClient instance is created, a call
    to the versioning API is made. This fixture makes sure that these calls succeed
    while not actually sending any requests to the API.

    It mocks:
    - VersioningApi.get_latest_pip_version to always return the current version. This
        avoids any errors/warnings related to not using the latest version.
    - VersioningApi.get_minimum_compatible_pip_version to always return 1.0.0 which
        should be compatible with all future versions.
    """

    def mock_get_latest_pip_version(current_version: str, **kwargs) -> str:
        return current_version

    # NOTE(guarin, 2/6/23): Cannot use pytest mocker fixture here because it has not
    # a "module" scope and it is not possible to use a fixture that has a tighter scope
    # inside a fixture with a wider scope.
    with mock.patch(
        "lightly.api._version_checking.VersioningApi.get_latest_pip_version",
        new=mock_get_latest_pip_version,
    ), mock.patch(
        "lightly.api._version_checking.VersioningApi.get_minimum_compatible_pip_version",
        return_value="1.0.0",
    ):
        yield


# The session start/finish pool hooks below are adapted from torchmetrics
# (Apache-2.0), tests/unittests/conftest.py:
# https://github.com/Lightning-AI/torchmetrics/blob/master/tests/unittests/conftest.py
def pytest_sessionstart():
    """Start the reusable gloo pool once per session when enabled. See #1982."""
    if not USE_PYTEST_POOL:
        return
    # Use spawn (as torchmetrics does): forking a process that has already
    # imported torch inherits its threads and deadlocks the workers.
    pool = mp.get_context("spawn").Pool(processes=NUM_PROCESSES)
    pool.starmap(setup_ddp, [(rank, NUM_PROCESSES) for rank in range(NUM_PROCESSES)])
    pytest.pool = pool


def pytest_sessionfinish():
    """Tear down the gloo pool at the end of the session. See #1982."""
    if not USE_PYTEST_POOL:
        return
    pytest.pool.starmap(
        teardown_ddp, [(rank, NUM_PROCESSES) for rank in range(NUM_PROCESSES)]
    )
    pytest.pool.close()
    pytest.pool.join()
