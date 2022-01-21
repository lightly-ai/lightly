import sys
import tempfile
import unittest

import numpy as np

from lightly.utils import save_custom_metadata
from lightly.utils.io import check_filenames, save_embeddings, check_embeddings
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup, MockedApiWorkflowClient


class TestCLICrop(MockedApiWorkflowSetup):

    @classmethod
    def setUpClass(cls) -> None:
        sys.modules["lightly.cli.upload_cli"].ApiWorkflowClient = MockedApiWorkflowClient

    def test_save_metadata(self):
        metadata = [("filename.jpg", {"random_metadata": 42})]
        metadata_filepath = tempfile.mktemp('.json', 'metadata')
        save_custom_metadata(metadata_filepath, metadata)

    def test_valid_filenames(self):
        valid = 'img.png'
        non_valid = 'img,1.png'
        filenames_list = [
            ([valid], True),
            ([valid, valid], True),
            ([non_valid], False),
            ([valid, non_valid], False),
        ]
        for filenames, valid in filenames_list:
            with self.subTest(msg=f"filenames:{filenames}"):
                if valid:
                    check_filenames(filenames)
                else:
                    with self.assertRaises(ValueError):
                        check_filenames(filenames)

class TestEmbeddingsIO(unittest.TestCase):

    def setUp(self):
        # correct embedding file as created through lightly
        self.embeddings_path = tempfile.mktemp('.csv', 'embeddings')
        embeddings = np.random.rand(32, 2)
        labels = [0 for i in range(len(embeddings))]
        filenames = [f'img_{i}.jpg' for i in range(len(embeddings))]
        save_embeddings(self.embeddings_path, embeddings, labels, filenames)

    def test_valid_embeddings(self):
        check_embeddings(self.embeddings_path)

    def test_whitespace_in_embeddings(self):
        # should fail because there whitespaces in the header columns
        lines = ['filenames, embedding_0,embedding_1,labels\n',
                 'img_1.jpg, 0.351,0.1231']
        with open(self.embeddings_path, 'w') as f:
            f.writelines(lines)
        with self.assertRaises(RuntimeError) as context:
            check_embeddings(self.embeddings_path)
        self.assertTrue('must not contain whitespaces' in str(context.exception))

    def test_no_labels_in_embeddings(self):
        # should fail because there is no `labels` column in the header
        lines = ['filenames,embedding_0,embedding_1\n',
                 'img_1.jpg,0.351,0.1231']
        with open(self.embeddings_path, 'w') as f:
            f.writelines(lines)
        with self.assertRaises(RuntimeError) as context:
            check_embeddings(self.embeddings_path)
        self.assertTrue('must end with `labels`' in str(context.exception))
