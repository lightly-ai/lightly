import os
import tempfile

import torchvision
from lightly.data.dataset import LightlyDataset

from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup


class TestApiWorkflowUploadDataset(MockedApiWorkflowSetup):
    def setUp(self) -> None:
        MockedApiWorkflowSetup.setUp(self)
        self.n_data = 100
        self.create_fake_dataset()
        self.api_workflow_client.tags_api.no_tags = 0

    def create_fake_dataset(self):
        n_data = self.n_data
        self.dataset = torchvision.datasets.FakeData(size=n_data,
                                                     image_size=(3, 32, 32))

        self.folder_path = tempfile.mkdtemp()
        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        self.sample_names = sample_names
        for sample_idx in range(n_data):
            data = self.dataset[sample_idx]
            path = os.path.join(self.folder_path, sample_names[sample_idx])
            data[0].save(path)

    def test_upload_dataset_over_quota(self):
        quota = self.n_data-1
        def get_quota_reduced():
            return str(quota)
        self.api_workflow_client.quota_api.get_quota = get_quota_reduced
        with self.assertRaises(ValueError):
            self.api_workflow_client.upload_dataset(input=self.folder_path)


    def test_upload_dataset_from_folder(self):
        self.api_workflow_client.upload_dataset(input=self.folder_path)

    def test_upload_existing_dataset(self):
        self.api_workflow_client.tags_api.no_tags = 2
        with self.assertWarns(Warning):
            self.api_workflow_client.upload_dataset(input=self.folder_path)

    def test_upload_dataset_from_dataset(self):
        dataset = LightlyDataset.from_torch_dataset(self.dataset)
        self.api_workflow_client.upload_dataset(input=dataset)
