from json import load
import os
import io
import csv
import random
import tempfile

import numpy as np
from lightly.utils.io import save_embeddings, load_embeddings, INVALID_FILENAME_CHARACTERS

import lightly
from tests.api_workflow.mocked_api_workflow_client import \
    MockedApiWorkflowSetup, N_FILES_ON_SERVER


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

        self.sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        if special_name_first_sample:
            self.sample_names[0] = "bliblablub"
        if special_char_in_first_filename:
            self.sample_names[0] = f'_{special_char_in_first_filename}' \
                              f'{self.sample_names[0]}'
        labels = [0] * len(self.sample_names)
        save_embeddings(
            self.path_to_embeddings,
            np.random.randn(n_data, n_dims),
            labels,
            self.sample_names
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
            special_char_in_first_filename=special_char_in_first_filename,
        )

        # perform the workflow to upload the embeddings
        self.api_workflow_client.upload_embeddings(path_to_embeddings_csv=self.path_to_embeddings, name="embedding_xyz")
        self.api_workflow_client.n_dims_embeddings_on_server = n_dims

    def test_upload_success(self):
        n_data = len(self.api_workflow_client.mappings_api.sample_names)
        self.t_ester_upload_embedding(n_data=n_data)
        filepath_embeddings_sorted = os.path.join(self.folder_path, "embeddings_sorted.csv")
        self.assertFalse(os.path.isfile(filepath_embeddings_sorted))

    def test_upload_wrong_length(self):
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

        # mock the embeddings on the server
        self.api_workflow_client.n_embedding_rows_on_server = N_FILES_ON_SERVER
        self.api_workflow_client.n_dims_embeddings_on_server = 32

        self.api_workflow_client.append_embeddings(
            self.path_to_embeddings,
            'embedding_id_xyz_2',
        )


    def test_append_embeddings_with_overlap(self):
    
        # first upload embeddings
        n_data = len(self.api_workflow_client.mappings_api.sample_names)
        self.t_ester_upload_embedding(n_data=n_data)

        # create a new set of embeddings overlapping with current embeddings
        self.create_fake_embeddings(100)

        # mock the embeddings on the server
        self.api_workflow_client.n_embedding_rows_on_server = N_FILES_ON_SERVER
        self.api_workflow_client.n_dims_embeddings_on_server = 32
        self.api_workflow_client.embeddings_filename_base = 'img'

        # the mock embeddings function returns embeddings which overlap with the ones generated above
        self.api_workflow_client.append_embeddings(
            self.path_to_embeddings,
            'embedding_id_xyz_2',
        )
        TestApiWorkflowUploadEmbeddings.EMBEDDINGS_FILENAME_BASE: str = 'sample'

        # load embeddings
        _, labels, filenames = load_embeddings(self.path_to_embeddings)

        # make sure the list of filenames is equal
        self.assertListEqual(
            sorted(self.sample_names),
            sorted(filenames),
        )

        # make sure that only new embeddings were added
        # all local labels are 0 (see "create_fake_embeddings") and all online
        # labels are the line, therefore the labels of the updated embeddings
        # must be 20 * [0] + [20, 11, 12, ..., 98, 99]
        self.assertListEqual(
            sorted(labels),
            sorted([0] * n_data + [i for i in range(n_data, 100)])
        )


    def test_append_embeddings_different_shape(self):

        # first upload embeddings
        n_data = len(self.api_workflow_client.mappings_api.sample_names)
        self.t_ester_upload_embedding(n_data=n_data)

        # create a new set of embeddings
        self.create_fake_embeddings(10, n_dims=16) # default is 32

        self.api_workflow_client.n_embedding_rows_on_server = N_FILES_ON_SERVER
        self.api_workflow_client.n_dims_embeddings_on_server = 32

        with self.assertRaises(RuntimeError):
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

