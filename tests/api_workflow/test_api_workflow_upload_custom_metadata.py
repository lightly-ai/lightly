import copy
import json
import os
import random
import tempfile
import pathlib
from typing import List

import numpy as np
import torchvision

from lightly.api.utils import MAXIMUM_FILENAME_LENGTH
from lightly.data.dataset import LightlyDataset
from lightly.openapi_generated.swagger_client import SampleData
from lightly.utils.io import COCO_ANNOTATION_KEYS

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
        coco_json[COCO_ANNOTATION_KEYS.images] = [{'id': i, 'file_name': fname} for i, fname in
                               enumerate(sample_names)]
        coco_json[COCO_ANNOTATION_KEYS.custom_metadata] = [{'id': i, 'image_id': i, 'custom_metadata': 0}
                                 for i, _ in enumerate(sample_names)]

        self.custom_metadata_file = tempfile.NamedTemporaryFile(mode="w+")
        json.dump(coco_json, self.custom_metadata_file)
        self.custom_metadata_file.flush()

    def test_upload_custom_metadata_one_step(self):
        self.create_fake_dataset()
        with open(self.custom_metadata_file.name, 'r') as f:
            custom_metadata = json.load(f)
            self.api_workflow_client.upload_dataset(input=self.folder_path, custom_metadata=custom_metadata)

    def test_upload_custom_metadata_two_steps_verbose(self):
        self.create_fake_dataset()
        self.api_workflow_client.upload_dataset(input=self.folder_path)
        with open(self.custom_metadata_file.name, 'r') as f:
            custom_metadata = json.load(f)
            self.api_workflow_client.upload_custom_metadata(custom_metadata, verbose=True)

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

    def test_upload_custom_metadata_with_append(self):
        self.create_fake_dataset()
        self.api_workflow_client.upload_dataset(input=self.folder_path)
        with open(self.custom_metadata_file.name, 'r') as f:
            custom_metadata = json.load(f)
            custom_metadata['metadata'] = custom_metadata['metadata'][:3]
            self.api_workflow_client.upload_custom_metadata(custom_metadata)


    def subtest_upload_custom_metadata(
            self,
            image_ids_images: List[int],
            image_ids_annotations: List[int],
            filenames_server: List[str]
    ):

        def get_samples_by_dataset_id(*args, **kwargs)-> List[SampleData]:
            samples = [
                SampleData(
                    id="dfd",
                    file_name=filename,
                    dataset_id='dataset_id_xyz',
                    type='Images'
                )
                for filename in filenames_server
            ]
            return samples
        self.api_workflow_client._samples_api.get_samples_by_dataset_id = get_samples_by_dataset_id
        filenames_metadata = [f"img_{id}.jpg" for id in image_ids_images]

        with self.subTest(image_ids_images=image_ids_images,
                image_ids_annotations=image_ids_annotations,
                filenames_server=filenames_server

        ):
            custom_metadata = {
                COCO_ANNOTATION_KEYS.images: [
                        {
                        COCO_ANNOTATION_KEYS.images_id: id,
                        COCO_ANNOTATION_KEYS.images_filename: filename}
                    for id, filename in zip(image_ids_images, filenames_metadata)
                ],
                COCO_ANNOTATION_KEYS.custom_metadata: [
                    {
                        COCO_ANNOTATION_KEYS.custom_metadata_image_id: id,
                        "any_key": "any_value"
                    }
                    for id in image_ids_annotations
                ],
            }
            custom_metadata_malformatted = \
                len(set(image_ids_annotations) - set(image_ids_images))
            metatadata_without_filenames_on_server = \
                len(set(filenames_metadata) - set(filenames_server))>0
            if custom_metadata_malformatted \
                    or metatadata_without_filenames_on_server:
                with self.assertRaises(RuntimeError):
                    self.api_workflow_client.upload_custom_metadata(
                        custom_metadata
                    )
            else:
                self.api_workflow_client.upload_custom_metadata(
                    custom_metadata
                )


    def test_upload_custom_metadata(self):
        potential_image_ids_images = [[0, 1, 2], [-1, 1], list(range(10))]
        potential_image_ids_annotations = potential_image_ids_images
        potential_filenames_server = [[f"img_{id}.jpg" for id in ids] for ids in potential_image_ids_images]

        self.create_fake_dataset()
        self.api_workflow_client.upload_dataset(input=self.folder_path)

        for image_ids_images in potential_image_ids_images:
            for image_ids_annotations in potential_image_ids_annotations:
                for filenames_server in potential_filenames_server:
                    self.subtest_upload_custom_metadata(
                        image_ids_images,
                        image_ids_annotations,
                        filenames_server
                    )



