import os
import re
import sys
import tempfile

import torchvision
from hydra.experimental import compose, initialize

import lightly
from tests.api_workflow.mocked_api_workflow_client import \
    MockedApiWorkflowSetup, MockedApiWorkflowClient, N_FILES_ON_SERVER


class TestCLIMagic(MockedApiWorkflowSetup):

    @classmethod
    def setUpClass(cls) -> None:
        sys.modules["lightly.cli.upload_cli"].ApiWorkflowClient = MockedApiWorkflowClient

    def setUp(self):
        MockedApiWorkflowSetup.setUp(self)
        self.create_fake_dataset()
        with initialize(config_path="../../lightly/cli/config", job_name="test_app"):
            self.cfg = compose(config_name="config", overrides=[
                "token='123'",
                f"input_dir={self.folder_path}",
                "trainer.max_epochs=0"
            ])

    def create_fake_dataset(self, filename_appendix: str = ''):
        n_data = len(self.api_workflow_client.get_filenames())
        self.dataset = torchvision.datasets.FakeData(size=n_data, image_size=(3, 32, 32))

        self.folder_path = tempfile.mkdtemp()
        sample_names = [f'img_{i}{filename_appendix}.jpg' for i in range(n_data)]
        self.sample_names = sample_names
        for sample_idx in range(n_data):
            data = self.dataset[sample_idx]
            path = os.path.join(self.folder_path, sample_names[sample_idx])
            data[0].save(path)

    def parse_cli_string(self, cli_words: str):
        cli_words = cli_words.replace("lightly-magic ", "")
        cli_words = re.split("=| ", cli_words)
        assert len(cli_words) % 2 == 0
        dict_keys = cli_words[0::2]
        dict_values = cli_words[1::2]
        for key, value in zip(dict_keys, dict_values):
            value = value.strip('\"')
            value = value.strip('\'')
            self.cfg[key] = value

    def test_parse_cli_string(self):
        cli_string = "lightly-magic dataset_id='XYZ' upload='thumbnails'"
        self.parse_cli_string(cli_string)
        assert self.cfg["dataset_id"] == 'XYZ'
        assert self.cfg["upload"] == 'thumbnails'

    def test_magic_new_dataset_name(self):
        MockedApiWorkflowClient.n_dims_embeddings_on_server = 32
        cli_string = "lightly-magic new_dataset_name='dataset_name_xyz'"
        self.parse_cli_string(cli_string)
        lightly.cli.lightly_cli(self.cfg)

    def test_magic_new_dataset_id(self):
        MockedApiWorkflowClient.n_dims_embeddings_on_server = 32
        cli_string = "lightly-magic dataset_id='dataset_id_xyz'"
        self.parse_cli_string(cli_string)
        lightly.cli.lightly_cli(self.cfg)

    def tearDown(self) -> None:
        for filename in ["embeddings.csv", "embeddings_sorted.csv"]:
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass



