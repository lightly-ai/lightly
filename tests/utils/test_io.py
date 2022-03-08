import csv
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
        self.assertTrue('has no `labels` column' in str(context.exception))

    def test_no_empty_rows_in_embeddings(self):
        # should fail because there are empty rows in the embeddings file
        lines = ['filenames,embedding_0,embedding_1,labels\n',
                 'img_1.jpg,0.351,0.1231\n\n'
                 'img_2.jpg,0.311,0.6231']
        with open(self.embeddings_path, 'w') as f:
            f.writelines(lines)
        with self.assertRaises(RuntimeError) as context:
            check_embeddings(self.embeddings_path)
        self.assertTrue('must not have empty rows' in str(context.exception))

    def test_embeddings_extra_rows(self):
        rows = [
            ['filenames', 'embedding_0', 'embedding_1', 'labels', 'selected',
             'masked'],
            ['image_0.jpg', '3.4', '0.23', '0', '1', '0'],
            ['image_1.jpg', '3.4', '0.23', '1', '0', '1']
        ]
        with open(self.embeddings_path, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(rows)

        check_embeddings(self.embeddings_path, remove_additional_columns=True)

        with open(self.embeddings_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row_read, row_original in zip(csv_reader, rows):
                self.assertListEqual(row_read, row_original[:-2])

    def test_embeddings_extra_rows_special_order(self):
        input_rows = [
            ['filenames', 'embedding_0', 'embedding_1', 'masked', 'labels', 'selected'],
            ['image_0.jpg', '3.4', '0.23', '0', '1', '0'],
            ['image_1.jpg', '3.4', '0.23', '1', '0', '1']
        ]
        correct_output_rows = [
            ['filenames', 'embedding_0', 'embedding_1', 'labels'],
            ['image_0.jpg', '3.4', '0.23', '1'],
            ['image_1.jpg', '3.4', '0.23', '0']
        ]
        with open(self.embeddings_path, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(input_rows)

        check_embeddings(self.embeddings_path, remove_additional_columns=True)

        with open(self.embeddings_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row_read, row_original in zip(csv_reader, correct_output_rows):
                self.assertListEqual(row_read, row_original)


