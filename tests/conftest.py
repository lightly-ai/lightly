# content of conftest.py

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


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


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
