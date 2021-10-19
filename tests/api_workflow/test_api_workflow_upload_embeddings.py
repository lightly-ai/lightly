import os
import io
import csv
import random
import tempfile

import numpy as np
from lightly.utils.io import save_embeddings, INVALID_FILENAME_CHARACTERS

import lightly
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup



def mock_get_embeddings_from_api(read_url, n_rows: int = 10, n_dims: int = 32):

    rows_csv = []
    for i in range(n_rows):
        row = [f'sample_{i}.png']
        for _ in range(n_dims):
            row.append(random.uniform(0, 1))
        row.append(random.randint(0, 5))
        rows_csv.append(row)

    # save the csv rows in a temporary in-memory string file
    # using a csv writer and then read them as bytes
    f = tempfile.SpooledTemporaryFile(mode="rw")
    writer = csv.writer(f)
    writer.writerows(rows_csv)
    f.seek(0)
    buffer = io.StringIO(f.read())
    reader = csv.reader(buffer)

    return reader


class TestApiWorkflowUploadEmbeddings(MockedApiWorkflowSetup):


    def create_fake_embeddings(self,
                               n_data,
                               n_dims: int = 32,
                               special_name_first_sample: bool = False,
                               special_char_in_first_filename: str = None):
        # create fake embeddings
        self.folder_path = tempfile.mkdtemp()
        self.path_to_embeddings = os.path.join(
            self.folder_path,
            'embeddings.csv'
        )
        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        if special_name_first_sample:
            sample_names[0] = "bliblablub"
        if special_char_in_first_filename:
            sample_names[0] = f'_{special_char_in_first_filename}' \
                              f'{sample_names[0]}'
        labels = [0] * len(sample_names)
        save_embeddings(
            self.path_to_embeddings,
            np.random.randn(n_data, n_dims),
            labels,
            sample_names
        )


    def t_ester_upload_embedding(self,
                                 n_data,
                                 n_dims: int = 32,
                                 special_name_first_sample: bool = False,
                                 special_char_in_first_filename: str = None):

        self.create_fake_embeddings(
            n_data,
            n_dims=n_dims,
            special_name_first_sample=special_name_first_sample,
            special_char_in_first_filename=special_char_in_first_filename
        )

        # perform the workflow to upload the embeddings
        self.api_workflow_client.upload_embeddings(path_to_embeddings_csv=self.path_to_embeddings, name="embedding_xyz")

    def test_upload_success(self):
        n_data = len(self.api_workflow_client.mappings_api.sample_names)
        self.t_ester_upload_embedding(n_data=n_data)
        filepath_embeddings_sorted = os.path.join(self.folder_path, "embeddings_sorted.csv")
        self.assertFalse(os.path.isfile(filepath_embeddings_sorted))

    def test_upload_wrong_lenght(self):
        n_data = 42 + len(self.api_workflow_client.mappings_api.sample_names)
        with self.assertRaises(ValueError):
            self.t_ester_upload_embedding(n_data=n_data)

    def test_upload_wrong_filenames(self):
        n_data = len(self.api_workflow_client.mappings_api.sample_names)
        with self.assertRaises(ValueError):
            self.t_ester_upload_embedding(n_data=n_data, special_name_first_sample=True)

    def test_upload_comma_filenames(self):
        n_data = len(self.api_workflow_client.mappings_api.sample_names)
        for invalid_char in INVALID_FILENAME_CHARACTERS:
            with self.subTest(msg=f"invalid_char: {invalid_char}"):
                with self.assertRaises(ValueError):
                    self.t_ester_upload_embedding(
                        n_data=n_data,
                        special_char_in_first_filename=invalid_char)

    def test_set_embedding_id_success(self):
        embedding_name = self.api_workflow_client.embeddings_api.embeddings[0].name
        self.api_workflow_client.set_embedding_id_by_name(embedding_name)

    def test_set_embedding_id_failure(self):
        embedding_name = "blibblabblub"
        with self.assertRaises(ValueError):
            self.api_workflow_client.set_embedding_id_by_name(embedding_name)

    def test_set_embedding_id_default(self):
        self.api_workflow_client.set_embedding_id_by_name()

    def test_append_embeddings(self):
    
        # first upload embeddings
        n_data = len(self.api_workflow_client.mappings_api.sample_names)
        self.t_ester_upload_embedding(n_data=n_data)

        # create a new set of embeddings
        self.create_fake_embeddings(10)

        lightly.api.api_workflow_upload_embeddings._get_csv_reader_from_read_url = mock_get_embeddings_from_api
        self.api_workflow_client.append_embeddings(
            self.path_to_embeddings,
            'embedding_id_xyz_2',
        )

    def test_append_embeddings_different_shape(self):

        # first upload embeddings
        n_data = len(self.api_workflow_client.mappings_api.sample_names)
        self.t_ester_upload_embedding(n_data=n_data)

        # create a new set of embeddings
        self.create_fake_embeddings(10, n_dims=16) # default is 32

        with self.assertRaises(RuntimeError):
            lightly.api.api_workflow_upload_embeddings._get_csv_reader_from_read_url = mock_get_embeddings_from_api
            self.api_workflow_client.append_embeddings(
                self.path_to_embeddings,
                'embedding_id_xyz_2',
            )


    def tearDown(self) -> None:
        for filename in ["embeddings.csv", "embeddings_sorted.csv"]:
            if hasattr(self, 'folder_path'):
                try:
                    filepath = os.path.join(self.folder_path, filename)
                    os.remove(filepath)
                except FileNotFoundError:
                    pass

