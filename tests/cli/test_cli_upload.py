import os
import re
import sys
import json
import tempfile

import numpy as np
import torchvision
from hydra.experimental import compose, initialize

import lightly
from lightly.utils import save_embeddings
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup, MockedApiWorkflowClient


class TestCLIUpload(MockedApiWorkflowSetup):

    @classmethod
    def setUpClass(cls) -> None:
        sys.modules["lightly.cli.upload_cli"].ApiWorkflowClient = MockedApiWorkflowClient

    def setUp(self):
        self.create_fake_dataset()
        with initialize(config_path="../../lightly/cli/config", job_name="test_app"):
            self.cfg = compose(config_name="config", overrides=["token='123'", f"input_dir={self.folder_path}"])

    def create_fake_dataset(self, n_data: int=5):
        self.dataset = torchvision.datasets.FakeData(size=n_data,
                                                     image_size=(3, 32, 32))

        self.folder_path = tempfile.mkdtemp()
        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        self.sample_names = sample_names
        for sample_idx in range(n_data):
            data = self.dataset[sample_idx]
            path = os.path.join(self.folder_path, sample_names[sample_idx])
            data[0].save(path)
        
        coco_json = {}
        coco_json['images'] = [
            {'id': i, 'file_name': fname} for i, fname in enumerate(self.sample_names)
        ]
        coco_json['metadata'] = [
            {'id': i, 'image_id': i, 'custom_metadata': 0 } for i, _ in enumerate(self.sample_names)
        ]
        
        self.tfile = tempfile.NamedTemporaryFile(mode="w+")
        json.dump(coco_json, self.tfile)
        self.tfile.flush()

        # create fake embeddings
        n_dims = 4
        self.path_to_embeddings = os.path.join(self.folder_path, 'embeddings.csv')
        labels = [0] * len(sample_names)
        save_embeddings(
            self.path_to_embeddings,
            np.random.randn(n_data, n_dims),
            labels,
            sample_names
        )


    def parse_cli_string(self, cli_words: str):
        cli_words = cli_words.replace("lightly-upload ", "")
        cli_words = re.split("=| ", cli_words)
        assert len(cli_words) % 2 == 0
        dict_keys = cli_words[0::2]
        dict_values = cli_words[1::2]
        for key, value in zip(dict_keys, dict_values):
            value = value.strip('\"')
            value = value.strip('\'')
            self.cfg[key] = value

    def test_parse_cli_string(self):
        cli_string = "lightly-upload dataset_id='XYZ' upload='thumbnails'"
        self.parse_cli_string(cli_string)
        assert self.cfg["dataset_id"] == 'XYZ'
        assert self.cfg["upload"] == 'thumbnails'

    def test_upload_no_token(self):
        self.cfg['token']=''
        with self.assertWarns(UserWarning):
            lightly.cli.upload_cli(self.cfg)

    def test_upload_new_dataset_name(self):
        cli_string = "lightly-upload new_dataset_name='new_dataset_name_xyz'"
        self.parse_cli_string(cli_string)
        lightly.cli.upload_cli(self.cfg)

    def test_upload_new_dataset_name_and_embeddings(self):
        cli_string = f"lightly-upload new_dataset_name='new_dataset_name_xyz' embeddings={self.path_to_embeddings}"
        self.parse_cli_string(cli_string)
        lightly.cli.upload_cli(self.cfg)

    def test_upload_new_dataset_id(self):
        cli_string = "lightly-upload dataset_id='xyz'"
        self.parse_cli_string(cli_string)
        lightly.cli.upload_cli(self.cfg)

    def test_upload_no_dataset(self):
        cli_string = "lightly-upload input_dir=data/ token='123'"
        self.parse_cli_string(cli_string)
        with self.assertWarns(UserWarning):
            lightly.cli.upload_cli(self.cfg)

    def test_upload_both_dataset(self):
        cli_string = "lightly-upload new_dataset_name='new_dataset_name_xyz' dataset_id='xyz'"
        self.parse_cli_string(cli_string)
        with self.assertWarns(UserWarning):
            lightly.cli.upload_cli(self.cfg)

    def test_upload_custom_metadata(self):
        cli_string = f"lightly-upload token='123' dataset_id='xyz' custom_metadata='{self.tfile.name}'"
        self.parse_cli_string(cli_string)
        lightly.cli.upload_cli(self.cfg)

    def test_upload_custom_metadata_only(self):
        cli_string = f"lightly-upload token='123' dataset_id='xyz' custom_metadata='{self.tfile.name}'"
        self.parse_cli_string(cli_string)
        input_dir = self.cfg['input_dir']
        self.cfg['input_dir'] = ''
        lightly.cli.upload_cli(self.cfg)
        self.cfg['input_dir'] = input_dir
