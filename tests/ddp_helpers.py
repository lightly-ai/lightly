"""Shared configuration for the distributed (DDP) test pool. See #1982.

Adapted from torchmetrics (Apache-2.0), which starts a pool of worker processes
once per session and reuses it for all distributed tests instead of spawning
fresh processes per test:
https://github.com/Lightning-AI/torchmetrics/blob/master/tests/unittests/conftest.py

Disabled by default; set ``USE_PYTEST_POOL=1`` to enable. The gloo backend and
non-Windows checks are folded into the flag so that a test's
``skipif(not USE_PYTEST_POOL)`` also covers platforms without a usable backend.

The pool assumes a single test session; running under pytest-xdist (``-n``)
would start one pool per worker, all contending for the same port.
"""

import os
import sys

import torch

USE_PYTEST_POOL = (
    os.getenv("USE_PYTEST_POOL", "0") == "1"
    and torch.distributed.is_available()
    and sys.platform not in ("win32", "cygwin")
)
NUM_PROCESSES = 2
# All ranks must rendezvous on the same port, so a single fixed port is used.
MASTER_PORT = "8088"


def setup_ddp(rank: int, world_size: int) -> None:
    """Initialize a gloo process group inside a pool worker. See #1982."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = MASTER_PORT
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
