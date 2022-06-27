import os
import shutil

from unittest.mock import patch

import PIL
import numpy as np

import torchvision

import lightly
from lightly.data.dataset import LightlyDataset

from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup
from lightly.openapi_generated.swagger_client.models.dataset_data import DatasetData



class TestApiWorkflowDownloadDataset(MockedApiWorkflowSetup):
    def setUp(self) -> None:
        MockedApiWorkflowSetup.setUp(self, dataset_id='dataset_0_id')
        self.api_workflow_client._tags_api.no_tags = 3

    def test_download_non_existing_tag(self):
        with self.assertRaises(ValueError):
            self.api_workflow_client.download_dataset('path/to/dir', tag_name='this_is_not_a_real_tag_name')

    def test_download_thumbnails(self):
        def get_thumbnail_dataset_by_id(*args):
            return DatasetData(name=f'dataset', id='dataset_id', last_modified_at=0,
                               type='thumbnails', size_in_bytes=-1, n_samples=-1, created_at=-1)
        self.api_workflow_client._datasets_api.get_dataset_by_id = get_thumbnail_dataset_by_id
        with self.assertRaises(ValueError):
            self.api_workflow_client.download_dataset('path/to/dir')

    def test_download_dataset(self):
        def my_func(read_url):
            return PIL.Image.fromarray(np.zeros((32, 32))).convert('RGB')
        #mock_get_image_from_readurl.return_value = PIL.Image.fromarray(np.zeros((32, 32)))
        lightly.api.api_workflow_download_dataset._get_image_from_read_url = my_func
        self.api_workflow_client.download_dataset('path-to-dir-remove-me', tag_name='initial-tag')
        shutil.rmtree('path-to-dir-remove-me')

    def test_export_label_box_data_rows_by_tag_name(self):
        rows = self.api_workflow_client.export_label_box_data_rows_by_tag_name('initial-tag')
        self.assertIsNotNone(rows)
        self.assertTrue(all(isinstance(row, dict) for row in rows))


    def test_export_label_studio_tasks_by_tag_name(self):
        tasks = self.api_workflow_client.export_label_studio_tasks_by_tag_name('initial-tag')
        self.assertIsNotNone(tasks)
        self.assertTrue(all(isinstance(task, dict) for task in tasks))

    def test_export_filenames_by_tag_name(self):
        filenames = self.api_workflow_client.export_filenames_by_tag_name('initial-tag')
        self.assertIsNotNone(filenames)
        self.assertTrue(isinstance(filenames, str))