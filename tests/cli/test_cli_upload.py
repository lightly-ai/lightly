import json
import os
import re
import sys
import tempfile
import warnings

import numpy as np
import torchvision
from hydra.experimental import compose, initialize

import lightly
from lightly.api.api_workflow_upload_embeddings import EmbeddingDoesNotExistError
from lightly.cli.upload_cli import SUCCESS_RETURN_VALUE
from lightly.openapi_generated.swagger_client import DatasetEmbeddingData
from lightly.utils.io import save_embeddings
from tests.api_workflow.mocked_api_workflow_client import (
    N_FILES_ON_SERVER,
    MockedApiWorkflowClient,
    MockedApiWorkflowSetup,
)


class TestCLIUpload(MockedApiWorkflowSetup):
    @classmethod
    def setUpClass(cls) -> None:
        sys.modules[
            "lightly.cli.upload_cli"
        ].ApiWorkflowClient = MockedApiWorkflowClient

    def set_tags(self, zero_tags: bool = True):
        # make the dataset appear empty
        def mocked_get_all_tags_zero(*args, **kwargs):
            if zero_tags:
                return []
            else:
                return ["Any tag"]

        MockedApiWorkflowClient.get_all_tags = mocked_get_all_tags_zero

    def set_embedding(self, has_embedding: bool):
        def mocked_get_embedding_by_name(*args, **kwargs):
            if has_embedding:
                return DatasetEmbeddingData(
                    id="embedding_id",
                    name="name",
                    is_processed=True,
                    created_at=0,
                )
            else:
                raise EmbeddingDoesNotExistError

        MockedApiWorkflowClient.get_embedding_by_name = mocked_get_embedding_by_name

    def setUp(self):
        # make the API dataset appear empty
        self.set_tags(zero_tags=True)
        # make it have no embeddings
        self.set_embedding(has_embedding=False)

        self.create_fake_dataset()

    def create_fake_dataset(
        self,
        n_data: int = N_FILES_ON_SERVER,
        n_rows_embeddings: int = N_FILES_ON_SERVER,
        n_dims_embeddings: int = 4,
    ):
        self.dataset = torchvision.datasets.FakeData(
            size=n_data, image_size=(3, 32, 32)
        )

        self.folder_path = tempfile.mkdtemp()
        sample_names = [f"img_{i}.jpg" for i in range(n_data)]
        self.sample_names = sample_names
        for sample_idx in range(n_data):
            data = self.dataset[sample_idx]
            path = os.path.join(self.folder_path, sample_names[sample_idx])
            data[0].save(path)

        coco_json = {}
        coco_json["images"] = [
            {"id": i, "file_name": fname} for i, fname in enumerate(self.sample_names)
        ]
        coco_json["metadata"] = [
            {"id": i, "image_id": i, "custom_metadata": 0}
            for i, _ in enumerate(self.sample_names)
        ]

        self.tfile = tempfile.NamedTemporaryFile(mode="w+")
        json.dump(coco_json, self.tfile)
        self.tfile.flush()

        # create fake embeddings
        self.path_to_embeddings = os.path.join(self.folder_path, "embeddings.csv")
        sample_names_embeddings = [f"img_{i}.jpg" for i in range(n_rows_embeddings)]
        labels = [0] * len(sample_names_embeddings)
        save_embeddings(
            self.path_to_embeddings,
            np.random.randn(n_rows_embeddings, n_dims_embeddings),
            labels,
            sample_names_embeddings,
        )
        MockedApiWorkflowClient.n_dims_embeddings_on_server = n_dims_embeddings
        MockedApiWorkflowClient.n_embedding_rows_on_server = n_rows_embeddings

    def parse_cli_string(
        self,
        cli_words: str,
    ):
        with initialize(config_path="../../lightly/cli/config", job_name="test_app"):
            overrides = [
                "token='123'",
                f"input_dir={self.folder_path}",
                f"embeddings={self.path_to_embeddings}",
            ]
            self.cfg = compose(config_name="config", overrides=overrides)

        sys.argv = re.split(" ", cli_words)
        self.cfg.merge_with_cli()

    def test_parse_cli_string(self):
        cli_string = (
            f"lightly-upload dataset_id='XYZ' upload='thumbnails' append={True}"
        )
        self.parse_cli_string(cli_string)
        self.assertEqual(self.cfg["dataset_id"], "XYZ")
        self.assertEqual(self.cfg["upload"], "thumbnails")
        self.assertTrue(self.cfg["append"])

    def test_upload_no_token(self):
        cli_string = f"lightly-upload"
        self.parse_cli_string(cli_string)
        self.cfg["token"] = ""
        with self.assertWarns(UserWarning):
            lightly.cli.upload_cli(self.cfg)

    def test_upload_new_dataset_name(self):
        cli_string = "lightly-upload new_dataset_name='new_dataset_name_xyz'"
        self.parse_cli_string(cli_string)
        result = lightly.cli.upload_cli(self.cfg)
        self.assertEqual(result, SUCCESS_RETURN_VALUE)
        self.assertGreater(
            len(
                os.getenv(
                    self.cfg["environment_variable_names"]["lightly_last_dataset_id"]
                )
            ),
            0,
        )

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
        for append in [True, False]:
            for n_dims_embeddings in dims_embeddings_options:
                for n_dims_embeddings_server in dims_embeddings_options:
                    with self.subTest(
                        append=append,
                        n_dims_embeddings=n_dims_embeddings,
                        n_dims_embeddings_server=n_dims_embeddings_server,
                    ):
                        self.create_fake_dataset(
                            n_data=N_FILES_ON_SERVER,
                            n_rows_embeddings=N_FILES_ON_SERVER,
                            n_dims_embeddings=n_dims_embeddings,
                        )
                        MockedApiWorkflowClient.n_embedding_rows_on_server = (
                            n_embedding_rows_on_server
                        )
                        MockedApiWorkflowClient.n_dims_embeddings_on_server = (
                            n_dims_embeddings_server
                        )
                        self.set_embedding(has_embedding=True)
                        cli_string = f"lightly-upload new_dataset_name='new_dataset_name_xyz' embeddings={self.path_to_embeddings} append={append}"
                        self.parse_cli_string(cli_string)
                        if n_dims_embeddings != n_dims_embeddings_server and append:
                            with self.assertRaises(RuntimeError):
                                lightly.cli.upload_cli(self.cfg)
                        elif not append:
                            with self.assertWarns(UserWarning):
                                lightly.cli.upload_cli(self.cfg)
                        else:
                            result = lightly.cli.upload_cli(self.cfg)
                            self.assertEqual(result, SUCCESS_RETURN_VALUE)

    def test_upload_new_dataset_id(self):
        cli_string = "lightly-upload dataset_id='xyz'"
        self.parse_cli_string(cli_string)
        result = lightly.cli.upload_cli(self.cfg)
        self.assertEqual(result, SUCCESS_RETURN_VALUE)

    def test_upload_no_dataset(self):
        cli_string = "lightly-upload input_dir=data/ token='123'"
        self.parse_cli_string(cli_string)
        with self.assertWarns(UserWarning):
            lightly.cli.upload_cli(self.cfg)

    def test_upload_both_dataset(self):
        cli_string = (
            "lightly-upload new_dataset_name='new_dataset_name_xyz' dataset_id='xyz'"
        )
        self.parse_cli_string(cli_string)
        with self.assertWarns(UserWarning):
            lightly.cli.upload_cli(self.cfg)

    def test_upload_custom_metadata(self):
        cli_string = f"lightly-upload token='123' dataset_id='xyz' custom_metadata='{self.tfile.name}'"
        self.parse_cli_string(cli_string)
        result = lightly.cli.upload_cli(self.cfg)
        self.assertEqual(result, SUCCESS_RETURN_VALUE)

    def check_upload_dataset_and_embedding(
        self,
        input_dir: bool,
        existing_dataset: bool,
        embeddings_path: bool,
        existing_embedding: bool,
        append: bool,
    ):
        with self.subTest(
            input_dir=input_dir,
            existing_dataset=existing_dataset,
            embeddings_path=embeddings_path,
            existing_embedding=existing_embedding,
            append=append,
        ):
            self.set_tags(zero_tags=not existing_dataset)
            self.set_embedding(has_embedding=existing_embedding)

            cli_string = f"lightly-upload dataset_id='xyz' append={append}"
            if not input_dir:
                cli_string += f" input_dir=''"
            if not embeddings_path:
                cli_string += f" embeddings=''"
            self.parse_cli_string(cli_string)

            if not append and existing_dataset and input_dir:
                with self.assertWarns(UserWarning):
                    lightly.cli.upload_cli(self.cfg)
            elif not append and existing_embedding and embeddings_path:
                with self.assertWarns(UserWarning):
                    lightly.cli.upload_cli(self.cfg)
            else:
                result = lightly.cli.upload_cli(self.cfg)
                self.assertEqual(result, SUCCESS_RETURN_VALUE)

    def test_upload_dataset_and_embedding(self):
        for input_dir in [True, False]:
            for existing_dataset in [True, False]:
                for embeddings_path in [True, False]:
                    for existing_embedding in [True, False]:
                        for append in [True, False]:
                            self.check_upload_dataset_and_embedding(
                                input_dir=input_dir,
                                existing_dataset=existing_dataset,
                                embeddings_path=embeddings_path,
                                existing_embedding=existing_embedding,
                                append=append,
                            )
