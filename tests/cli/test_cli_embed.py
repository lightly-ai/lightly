import os
import re
import sys
import tempfile

import torchvision
from hydra.experimental import compose, initialize

import lightly
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup, MockedApiWorkflowClient


class TestCLIEmbed(MockedApiWorkflowSetup):

    @classmethod
    def setUpClass(cls) -> None:
        sys.modules["lightly.cli.embed_cli"].ApiWorkflowClient = \
            MockedApiWorkflowClient

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
        n_data = 16
        self.dataset = torchvision.datasets.FakeData(size=n_data, image_size=(3, 32, 32))

        self.folder_path = tempfile.mkdtemp()
        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        self.sample_names = sample_names
        for sample_idx in range(n_data):
            data = self.dataset[sample_idx]
            path = os.path.join(self.folder_path, sample_names[sample_idx])
            data[0].save(path)

    def test_embed(self):
        lightly.cli.embed_cli(self.cfg)

    def tearDown(self) -> None:
        for filename in ["embeddings.csv", "embeddings_sorted.csv"]:
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass



