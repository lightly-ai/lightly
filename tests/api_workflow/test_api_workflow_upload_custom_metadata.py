import copy
import json
import os
import random
import tempfile
import pathlib

import numpy as np
import torchvision

from lightly.api.utils import MAXIMUM_FILENAME_LENGTH
from lightly.data.dataset import LightlyDataset

from tests.api_workflow.mocked_api_workflow_client import \
    MockedApiWorkflowSetup

import cv2


class TestApiWorkflowUploadCustomMetadata(MockedApiWorkflowSetup):

    def create_fake_dataset(self, n_data: int = 10, sample_names=None):
        self.dataset = torchvision.datasets.FakeData(size=n_data,
                                                     image_size=(3, 32, 32))

        self.folder_path = tempfile.mkdtemp()
        image_extension = '.jpg'
        sample_names = sample_names if sample_names is not None else [
            f'img_{i}{image_extension}' for i in range(n_data)]
        for sample_idx in range(n_data):
            data = self.dataset[sample_idx]
            sample_name = sample_names[sample_idx]
            path = os.path.join(self.folder_path, sample_name)
            data[0].save(path)

        coco_json = dict()
        coco_json['images'] = [{'id': i, 'file_name': fname} for i, fname in
                               enumerate(sample_names)]
        coco_json['metadata'] = [{'id': i, 'image_id': i, 'custom_metadata': 0}
                                 for i, _ in enumerate(sample_names)]

        self.custom_metadata_file = tempfile.NamedTemporaryFile(mode="w+")
        json.dump(coco_json, self.custom_metadata_file)
        self.custom_metadata_file.flush()

    def test_upload_custom_metadata_one_step(self):
        self.create_fake_dataset()
        with open(self.custom_metadata_file.name, 'r') as f:
            custom_metadata = json.load(f)
            self.api_workflow_client.upload_dataset(input=self.folder_path, custom_metadata=custom_metadata)

    def test_upload_custom_metadata_two_steps(self):
        self.create_fake_dataset()
        self.api_workflow_client.upload_dataset(input=self.folder_path)
        with open(self.custom_metadata_file.name, 'r') as f:
            custom_metadata = json.load(f)
            self.api_workflow_client.upload_custom_metadata(custom_metadata)

    def test_upload_custom_metadata_before_uploading_samples(self):
        self.create_fake_dataset()
        with open(self.custom_metadata_file.name, 'r') as f:
            custom_metadata = json.load(f)
            with self.assertRaises(ValueError):
                self.api_workflow_client.upload_custom_metadata(custom_metadata)
