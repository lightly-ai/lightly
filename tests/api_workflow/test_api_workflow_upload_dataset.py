import copy
import os
import random
import tempfile
import pathlib

import numpy as np
import torchvision

from lightly.api.utils import MAXIMUM_FILENAME_LENGTH
from lightly.data.dataset import LightlyDataset

from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup

import cv2


class TestApiWorkflowUploadDataset(MockedApiWorkflowSetup):
    def setUp(self) -> None:
        MockedApiWorkflowSetup.setUp(self)
        self.n_data = 100
        self.create_fake_dataset()
        self.api_workflow_client.tags_api.no_tags = 0

    def create_fake_dataset(self, length_of_filepath: int = -1, sample_names=None):
        n_data = self.n_data if sample_names is None else len(sample_names)
        self.dataset = torchvision.datasets.FakeData(size=n_data,
                                                     image_size=(3, 32, 32))

        self.folder_path = tempfile.mkdtemp()
        image_extension = '.jpg'
        sample_names = sample_names if sample_names is not None else [f'img_{i}{image_extension}' for i in range(n_data)]
        for sample_idx in range(n_data):
            data = self.dataset[sample_idx]
            sample_name = sample_names[sample_idx]
            path = os.path.join(self.folder_path, sample_name)

            if length_of_filepath > len(path):
                assert path.endswith(image_extension)
                n_missing_chars = length_of_filepath - len(path)
                path = path[:-len(image_extension)] + 'x' * n_missing_chars + image_extension

            data[0].save(path)

    def corrupt_fake_dataset(self):
        n_data = self.n_data
        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        for sample_name in sample_names:
            pathlib.Path(os.path.join(self.folder_path, sample_name)).touch()

    def test_upload_dataset_no_dataset(self):
        with self.assertRaises(ValueError):
            self.api_workflow_client.upload_dataset(1)

    def test_upload_dataset_over_quota(self):
        quota = self.n_data - 1

        def get_quota_reduced():
            return str(quota)

        self.api_workflow_client.quota_api.get_quota_maximum_dataset_size = get_quota_reduced
        with self.assertRaises(ValueError):
            self.api_workflow_client.upload_dataset(input=self.folder_path)

    def test_upload_dataset_from_folder(self):
        self.api_workflow_client.upload_dataset(input=self.folder_path)

    def test_upload_dataset_from_folder_full(self):
        self.api_workflow_client.upload_dataset(input=self.folder_path, mode="full")

    def test_upload_dataset_from_folder_only_metadata(self):
        self.api_workflow_client.upload_dataset(input=self.folder_path, mode="metadata")

    def test_upsize_existing_dataset(self):
        self.api_workflow_client.tags_api.no_tags = 1
        self.api_workflow_client.upload_dataset(input=self.folder_path)

    def test_upload_dataset_from_dataset(self):
        dataset = LightlyDataset.from_torch_dataset(self.dataset)
        self.api_workflow_client.upload_dataset(input=dataset)

    def test_corrupt_dataset_from_folder(self):
        self.corrupt_fake_dataset()
        self.api_workflow_client.upload_dataset(input=self.folder_path)
        self.api_workflow_client.upload_dataset(input=self.folder_path)

    def test_filename_length_lower(self):
        self.create_fake_dataset(length_of_filepath=MAXIMUM_FILENAME_LENGTH - 1)
        self.api_workflow_client.upload_dataset(input=self.folder_path)

        samples = self.api_workflow_client.samples_api.get_samples_by_dataset_id(dataset_id="does not matter")
        self.assertEqual(self.n_data, len(samples))

    def test_filename_length_upper(self):
        self.create_fake_dataset(length_of_filepath=MAXIMUM_FILENAME_LENGTH + 10)
        self.api_workflow_client.upload_dataset(input=self.folder_path)

        samples = self.api_workflow_client.samples_api.get_samples_by_dataset_id(dataset_id="does not matter")
        self.assertEqual(0, len(samples))

    def create_fake_video_dataset(self, n_videos=5, n_frames_per_video=10, w=32, h=32, c=3, extension='avi'):

        self.video_input_dir = tempfile.mkdtemp()
        self.frames = (np.random.randn(n_frames_per_video, w, h, c) * 255).astype(np.uint8)

        for i in range(n_videos):
            path = os.path.join(self.video_input_dir, f'output-{i}.{extension}')
            out = cv2.VideoWriter(path, 0, 1, (w, h))
            for frame in self.frames:
                out.write(frame)
            out.release()

    def test_upload_video_dataset_from_folder(self):
        self.create_fake_video_dataset()
        self.api_workflow_client.upload_dataset(input=self.folder_path)

    def test_upload_dataset_twice(self):
        rng = np.random.default_rng(2021)

        base_upload_single_image = self.api_workflow_client._upload_single_image

        # Upload with some uploads failing
        def failing_upload_sample(*args, **kwargs):
            if rng.random() < 0.9:
                return base_upload_single_image(*args, **kwargs)
            else:
                raise ValueError()

        self.api_workflow_client._upload_single_image = failing_upload_sample
        self.api_workflow_client.upload_dataset(input=self.folder_path)

        # Ensure that not all samples were uploaded
        samples = self.api_workflow_client.samples_api.get_samples_by_dataset_id(dataset_id="does not matter")
        self.assertLess(len(samples), self.n_data)

        # Upload without failing uploads
        self.api_workflow_client._upload_single_image = base_upload_single_image
        self.api_workflow_client.upload_dataset(input=self.folder_path)

        # Ensure that now all samples were uploaded exactly once
        samples = self.api_workflow_client.samples_api.get_samples_by_dataset_id(dataset_id="does not matter")
        self.assertEqual(self.n_data, len(samples))


    def test_upload_dataset_twice_with_overlap(self):

        all_sample_names = [f'img_upload_twice_{i}.jpg' for i in range(10)]

        # upload first part of the dataset (sample_0 - sample_6)
        self.create_fake_dataset(sample_names=all_sample_names[:7])
        self.api_workflow_client.upload_dataset(input=self.folder_path)

        # upload second part of the dataset (sample_3 - sample_9)
        self.create_fake_dataset(sample_names=all_sample_names[3:])
        self.api_workflow_client.upload_dataset(input=self.folder_path)

        # always returns all samples so dataset_id doesn't matter
        samples = self.api_workflow_client.samples_api.get_samples_by_dataset_id(dataset_id='')

        # assert the filenames are the same
        self.assertListEqual(
            sorted(all_sample_names), 
            sorted([s.file_name for s in samples]),
        )
