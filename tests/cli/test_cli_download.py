import os
import sys
import tempfile

import pytest
import torchvision
from hydra.experimental import compose, initialize

import lightly
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowClient


class TestCLIDownload(MockedApiWorkflowSetup):

    @classmethod
    def setUpClass(cls) -> None:
        sys.modules["lightly.cli.download_cli"].ApiWorkflowClient = MockedApiWorkflowClient

    def setUp(self):
        with initialize(config_path="../../lightly/cli/config", job_name="test_app"):
            self.cfg = compose(config_name="config")

    def create_fake_dataset(self, n_data: int = 5):
        self.dataset = torchvision.datasets.FakeData(size=n_data,
                                                     image_size=(3, 32, 32))

        self.input_dir = tempfile.mkdtemp()

        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        self.sample_names = sample_names
        for sample_idx in range(n_data):
            data = self.dataset[sample_idx]
            path = os.path.join(self.input_dir, sample_names[sample_idx])
            data[0].save(path)

        self.output_dir = tempfile.mkdtemp()

    def parse_cli_string(self, cli_words: str):
        cli_words = cli_words.replace('lightly-download ', '')
        overrides = cli_words.split(' ')
        with initialize(config_path='../../lightly/cli/config/'):
            self.cfg = compose(
                config_name='config',
                overrides=overrides,
            )

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

    def test_download_no_tag_name(self):
        # defaults to initial-tag
        cli_string = "lightly-download token='123' dataset_id='XYZ'"
        self.parse_cli_string(cli_string)
        lightly.cli.download_cli(self.cfg)

    def test_download_no_token(self):
        cli_string = "lightly-download dataset_id='XYZ' tag_name='sampled_tag_xyz'"
        self.parse_cli_string(cli_string)
        with self.assertWarns(UserWarning):
            lightly.cli.download_cli(self.cfg)

    def test_download_no_dataset_id(self):
        cli_string = "lightly-download token='123' tag_name='sampled_tag_xyz'"
        self.parse_cli_string(cli_string)
        with self.assertWarns(UserWarning):
            lightly.cli.download_cli(self.cfg)

    def test_download_copy_from_input_to_output_dir(self):
        self.create_fake_dataset(n_data=100)
        cli_string = f"lightly-download token='123' dataset_id='dataset_1_id' tag_name='sampled_tag_xyz' " \
                     f"input_dir={self.input_dir} output_dir={self.output_dir}"
        self.parse_cli_string(cli_string)
        lightly.cli.download_cli(self.cfg)

    def test_download_from_tag_with_integer_name(self):
        """Test to reproduce issue #575."""
        # use tag name "1000"
        cli_string = "lightly-download token='123' dataset_id='dataset_1_id' tag_name=1000"
        self.parse_cli_string(cli_string)
        with pytest.warns(None) as record:
            lightly.cli.download_cli(self.cfg)
        # check if the warning "Tag with name 1000 does not exist" is raised
        # if so, the cli string was not parsed correctly
        # (i.e. as int instead of str)
        self.assertEqual(len(record), 0)

    def tearDown(self) -> None:
        try:
            os.remove(f"{self.cfg['tag_name']}.txt")
        except FileNotFoundError:
            pass
