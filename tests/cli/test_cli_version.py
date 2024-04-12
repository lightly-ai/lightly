import os
import re
import sys
import tempfile

import hydra
import pytest
from hydra.experimental import compose

try:
    from hydra import initialize
except ImportError:
    from hydra.experimental import initialize

from lightly.cli.version_cli import version_cli
from tests.api_workflow.mocked_api_workflow_client import (
    MockedApiWorkflowClient,
    MockedApiWorkflowSetup,
)


class TestCLIVersion(MockedApiWorkflowSetup):
    def setUp(self):
        MockedApiWorkflowSetup.setUp(self)
        with initialize(config_path="../../lightly/cli/config", job_name="test_app"):
            self.cfg = compose(config_name="config")

    @pytest.fixture(autouse=True)
    def capsys(self, capsys):
        self.capsys = capsys

    def test_checkpoint_created(self):
        version_cli(self.cfg)
        out, err = self.capsys.readouterr()
        assert out.startswith("lightly version")
