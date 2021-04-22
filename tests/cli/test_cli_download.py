import os
import re
import sys
import tempfile

from hydra.experimental import compose, initialize

import lightly
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup, MockedApiWorkflowClient


#in download_cli.py: from lightly.api.api_workflow_client import ApiWorkflowClient

class TestCLIDownload(MockedApiWorkflowSetup):

    @classmethod
    def setUpClass(cls) -> None:
        sys.modules["lightly.cli.download_cli"].ApiWorkflowClient = MockedApiWorkflowClient


    def setUp(self):
        with initialize(config_path="../../lightly/cli/config", job_name="test_app"):
            self.cfg = compose(config_name="config", overrides=["token='123'", "dataset_id='XYZ'"])


    def parse_cli_string(self, cli_words: str):
        cli_words = cli_words.replace("lightly-download ", "")
        cli_words = re.split("=| ", cli_words)
        assert len(cli_words) % 2 == 0
        dict_keys = cli_words[0::2]
        dict_values = cli_words[1::2]
        for key, value in zip(dict_keys, dict_values):
            value = value.strip('\"')
            value = value.strip('\'')
            self.cfg[key] = value

    def test_parse_cli_string(self):
        cli_string = "lightly-download token='123' dataset_id='XYZ'"
        self.parse_cli_string(cli_string)
        assert self.cfg["token"] == '123'
        assert self.cfg["dataset_id"] == 'XYZ'

    def test_download_base(self):
        cli_string = "lightly-download token='123' dataset_id='XYZ'"
        self.parse_cli_string(cli_string)
        lightly.cli.download_cli(self.cfg)

    def test_download_tag_name(self):
        cli_string = "lightly-download token='123' dataset_id='XYZ' tag_name='sampled_tag_xyz'"
        self.parse_cli_string(cli_string)
        lightly.cli.download_cli(self.cfg)

    def test_download_tag_name_nonexisting(self):
        cli_string = "lightly-download token='123' dataset_id='XYZ' tag_name='nonexisting_xyz'"
        self.parse_cli_string(cli_string)
        with self.assertWarns(Warning):
            lightly.cli.download_cli(self.cfg)

    def test_download_tag_name_exclude_parent(self):
        cli_string = "lightly-download token='123' dataset_id='XYZ' tag_name='sampled_tag_xyz' exclude_parent_tag=True"
        self.parse_cli_string(cli_string)
        lightly.cli.download_cli(self.cfg)

    def tearDown(self) -> None:
        try:
            os.remove(f"{self.cfg['tag_name']}.txt")
        except FileNotFoundError:
            pass


