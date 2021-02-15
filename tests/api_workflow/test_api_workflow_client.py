import os
import tempfile
import unittest

import numpy as np

import lightly
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowClient


class TestApiWorkflow(unittest.TestCase):

    def test_upload_embedding(self, n_data: int = 100):
        # create fake embeddings
        folder_path = tempfile.mkdtemp()
        path_to_embeddings = os.path.join(
            folder_path,
            'embeddings.csv'
        )
        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        labels = [0] * len(sample_names)
        lightly.utils.save_embeddings(
            path_to_embeddings,
            np.random.randn(n_data, 16),
            labels,
            sample_names
        )

        # Set the workflow with mocked functions
        api_workflow_client = MockedApiWorkflowClient(host="host_xyz", token="token_xyz", dataset_id="dataset_id_xyz")

        # perform the workflow to upload the embeddings
        api_workflow_client.upload_embeddings(path_to_embeddings_csv=path_to_embeddings)