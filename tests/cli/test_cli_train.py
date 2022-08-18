import os
import re
import sys
import tempfile

import torchvision
from hydra.experimental import compose, initialize

import lightly
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup, MockedApiWorkflowClient


class TestCLITrain(MockedApiWorkflowSetup):

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

    def create_fake_dataset(self):
        n_data = 5
        self.dataset = torchvision.datasets.FakeData(size=n_data, image_size=(3, 32, 32))

        self.folder_path = tempfile.mkdtemp()
        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        self.sample_names = sample_names
        for sample_idx in range(n_data):
            data = self.dataset[sample_idx]
            path = os.path.join(self.folder_path, sample_names[sample_idx])
            data[0].save(path)

    def parse_cli_string(self, cli_words: str):
        cli_words = cli_words.replace("lightly-train ", "")
        cli_words = re.split("=| ", cli_words)
        assert len(cli_words) % 2 == 0
        dict_keys = cli_words[0::2]
        dict_values = cli_words[1::2]
        for key, value in zip(dict_keys, dict_values):
            value = value.strip('\"')
            value = value.strip('\'')
            key_parts = key.split(".")
            if len(key_parts) == 1:
                self.cfg[key_parts[0]]= value
            elif len(key_parts) == 2:
                self.cfg[key_parts[0]][key_parts[1]] = value
            else:
                raise ValueError

    def test_parse_cli_string(self):
        cli_string = "lightly-train trainer.weights_summary=top"
        self.parse_cli_string(cli_string)
        # TODO MICHAL
        assert self.cfg["trainer"]["weights_summary"] == 'top'

    def test_train_weights_summary(self):
        # TODO MICHAL
        for weights_summary in ["None", "top", "full"]:
            cli_string = f"lightly-train trainer.weights_summary={weights_summary}"
            with self.subTest(cli_string):
                self.parse_cli_string(cli_string)
                lightly.cli.train_cli(self.cfg)

                self.assertGreater(len(os.getenv(
                    self.cfg['environment_variable_names'][
                        'lightly_last_checkpoint_path'])), 0)

    def tearDown(self) -> None:
        for filename in ["embeddings.csv", "embeddings_sorted.csv"]:
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass
