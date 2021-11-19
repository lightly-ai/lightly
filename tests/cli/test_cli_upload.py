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
from tests.api_workflow.mocked_api_workflow_client import \
    MockedApiWorkflowSetup, MockedApiWorkflowClient, N_FILES_ON_SERVER


class TestCLIUpload(MockedApiWorkflowSetup):

    @classmethod
    def setUpClass(cls) -> None:
        sys.modules["lightly.cli.upload_cli"].ApiWorkflowClient = MockedApiWorkflowClient

    def setUp(self):
        self.create_fake_dataset()
        with initialize(config_path="../../lightly/cli/config", job_name="test_app"):
            self.cfg = compose(config_name="config", overrides=["token='123'", f"input_dir={self.folder_path}"])


    def create_fake_dataset(self, n_data: int=5, n_rows_embeddings: int=5, n_dims_embeddings: int = 4):
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
        self.path_to_embeddings = os.path.join(self.folder_path, 'embeddings.csv')
        sample_names_embeddings = [f'img_{i}.jpg' for i in range(n_rows_embeddings)]
        labels = [0] * len(sample_names_embeddings)
        save_embeddings(
            self.path_to_embeddings,
            np.random.randn(n_rows_embeddings, n_dims_embeddings),
            labels,
            sample_names_embeddings
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
        """
        Idea of workflow:
        We have 80 embedding rows on the server (n_embedding_rows_on_server).
        We have 100 filenames on the server (N_FILES_ON_SERVER).
        We have a dataset with 100 samples and 100 rows in the embeddings file.
        Then we upload the dataset -> the 20 new samples get uploaded,
        the 80 existing samples are skipped.
        The 80 embeddings on the server are tried to be added
        to the local embeddings file, but the local one already contains all
        these embedding rows. Thus the new file after the appending equals
        the local file before appending.

        """
        dims_embeddings_options = [8, 32]
        n_embedding_rows_on_server = 80
        for n_dims_embeddings in dims_embeddings_options:
            for n_dims_embeddings_server in dims_embeddings_options:
                with self.subTest(
                        f"test_{n_dims_embeddings}_{n_dims_embeddings_server}"
                ):

                    self.create_fake_dataset(
                        n_data=N_FILES_ON_SERVER,
                        n_rows_embeddings=N_FILES_ON_SERVER,
                        n_dims_embeddings=n_dims_embeddings
                    )
                    MockedApiWorkflowClient.n_embedding_rows_on_server = n_embedding_rows_on_server
                    MockedApiWorkflowClient.n_dims_embeddings_on_server = n_dims_embeddings_server
                    cli_string = f"lightly-upload new_dataset_name='new_dataset_name_xyz' embeddings={self.path_to_embeddings}"
                    self.parse_cli_string(cli_string)
                    if n_dims_embeddings != n_dims_embeddings_server:
                        with self.assertRaises(RuntimeError):
                            lightly.cli.upload_cli(self.cfg)
                    else:
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
