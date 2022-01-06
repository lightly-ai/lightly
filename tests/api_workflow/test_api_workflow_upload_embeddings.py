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
                               n_data_start: int = 0,
                               n_dims: int = 32,
                               special_name_first_sample: bool = False,
                               special_char_in_first_filename: str = None):
        # create fake embeddings
        self.folder_path = tempfile.mkdtemp()
        self.path_to_embeddings = os.path.join(
            self.folder_path,
            'embeddings.csv'
        )

        self.sample_names = [f'img_{i}.jpg' for i in range(n_data_start, n_data_start + n_data)]
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
                                 special_char_in_first_filename: str = None,
                                 name: str = "embedding_xyz"
                                 ):

        self.create_fake_embeddings(
            n_data,
            n_dims=n_dims,
            special_name_first_sample=special_name_first_sample,
            special_char_in_first_filename=special_char_in_first_filename,
        )

        # perform the workflow to upload the embeddings
        self.api_workflow_client.upload_embeddings(path_to_embeddings_csv=self.path_to_embeddings, name=name)
        self.api_workflow_client.n_dims_embeddings_on_server = n_dims

    def test_upload_success(self):
        n_data = len(self.api_workflow_client._mappings_api.sample_names)
        self.t_ester_upload_embedding(n_data=n_data)
        filepath_embeddings_sorted = os.path.join(self.folder_path, "embeddings_sorted.csv")
        self.assertFalse(os.path.isfile(filepath_embeddings_sorted))

    def test_upload_wrong_length(self):
        n_data = 42 + len(self.api_workflow_client._mappings_api.sample_names)
        with self.assertRaises(ValueError):
            self.t_ester_upload_embedding(n_data=n_data)

    def test_upload_wrong_filenames(self):
        n_data = len(self.api_workflow_client._mappings_api.sample_names)
        with self.assertRaises(ValueError):
            self.t_ester_upload_embedding(n_data=n_data, special_name_first_sample=True)

    def test_upload_comma_filenames(self):
        n_data = len(self.api_workflow_client._mappings_api.sample_names)
        for invalid_char in INVALID_FILENAME_CHARACTERS:
            with self.subTest(msg=f"invalid_char: {invalid_char}"):
                with self.assertRaises(ValueError):
                    self.t_ester_upload_embedding(
                        n_data=n_data,
                        special_char_in_first_filename=invalid_char)

    def test_set_embedding_id_default(self):
        self.api_workflow_client.set_embedding_id_to_latest()

    def test_upload_existing_embedding(self):
    
        # first upload embeddings
        n_data = len(self.api_workflow_client._mappings_api.sample_names)
        self.t_ester_upload_embedding(n_data=n_data)

        # create a new set of embeddings
        self.create_fake_embeddings(10)

        # mock the embeddings on the server
        self.api_workflow_client.n_dims_embeddings_on_server = 32

        self.api_workflow_client.append_embeddings(
            self.path_to_embeddings,
            'embedding_id_xyz_2',
        )

    def test_append_embeddings_with_overlap(self):

        # mock the embeddings on the server
        n_data_server = len(self.api_workflow_client._mappings_api.sample_names)
        self.api_workflow_client.n_dims_embeddings_on_server = 32

        # create new local embeddings overlapping with server embeddings
        n_data_start_local = n_data_server // 3
        n_data_local = n_data_server * 2
        self.create_fake_embeddings(n_data=n_data_local, n_data_start=n_data_start_local)

        """
        Assumptions:
            n_data_server = 100
            n_data_start_local = 33
            n_data_local = 200
        
        Server embeddings file:
            filenames: 0 ... 99
            labels: 0 ... 99
            
        Local embeddings file:
            filenames: 33 ... 232
            labels: 0 ... 0 (all zero)
            
        Appended embedding file must thus be:
            filenames: 0 ... 232
            labels: 0 ... 32 (from server) + 0 ... 0 (from local)
        """

        # append the local embeddings to the server embeddings
        self.api_workflow_client.append_embeddings(
            self.path_to_embeddings,
            'embedding_id_xyz_2',
        )

        # load the new (appended) embeddings
        _, labels_appended, filenames_appended = \
            load_embeddings(self.path_to_embeddings)

        # define the expected filenames and labels
        self.create_fake_embeddings(n_data=n_data_local + n_data_start_local)
        _, _, filenames_expected = load_embeddings(self.path_to_embeddings)
        labels_expected = list(range(n_data_start_local)) + [0] * n_data_local

        # make sure the list of filenames and labels equal
        self.assertListEqual(filenames_appended, filenames_expected)
        self.assertListEqual(labels_appended, labels_expected)


    def test_append_embeddings_different_shape(self):

        # first upload embeddings
        n_data = len(self.api_workflow_client._mappings_api.sample_names)
        self.t_ester_upload_embedding(n_data=n_data)

        # create a new set of embeddings
        self.create_fake_embeddings(10, n_dims=16) # default is 32

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

