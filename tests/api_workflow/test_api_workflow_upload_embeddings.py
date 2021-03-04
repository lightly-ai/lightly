import os
import tempfile

import numpy as np

import lightly
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup


class TestApiWorkflowUploadEmbeddigns(MockedApiWorkflowSetup):
    def t_ester_upload_embedding(self, n_data, special_name_first_sample: bool = False):
        # create fake embeddings
        folder_path = tempfile.mkdtemp()
        path_to_embeddings = os.path.join(
            folder_path,
            'embeddings.csv'
        )
        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        if special_name_first_sample:
            sample_names[0] = "bliblablub"
        labels = [0] * len(sample_names)
        lightly.utils.save_embeddings(
            path_to_embeddings,
            np.random.randn(n_data, 16),
            labels,
            sample_names
        )

        # perform the workflow to upload the embeddings
        self.api_workflow_client.upload_embeddings(path_to_embeddings_csv=path_to_embeddings, name="embedding_xyz")

    def test_upload_success(self):
        n_data = len(self.api_workflow_client.mappings_api.sample_names)
        self.t_ester_upload_embedding(n_data=n_data)

    def test_upload_wrong_lenght(self):
        n_data = 42 + len(self.api_workflow_client.mappings_api.sample_names)
        with self.assertRaises(ValueError):
            self.t_ester_upload_embedding(n_data=n_data)

    def test_upload_wrong_filenames(self):
        n_data = len(self.api_workflow_client.mappings_api.sample_names)
        with self.assertRaises(ValueError):
            self.t_ester_upload_embedding(n_data=n_data, special_name_first_sample=True)
