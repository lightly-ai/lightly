import os
import tempfile

import torchvision
from lightly.data.dataset import LightlyDataset

from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup
from lightly.openapi_generated.swagger_client.models.dataset_data import DatasetData


class TestApiWorkflowDownloadDataset(MockedApiWorkflowSetup):
    def setUp(self) -> None:
        MockedApiWorkflowSetup.setUp(self, dataset_id='dataset_0_id')
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

    def test_download_non_existing_tag(self):
        with self.assertRaises(ValueError):
            self.api_workflow_client.download_dataset('path/to/dir', tag_name='this_is_not_a_real_tag_name')

    def test_download_thumbnails(self):
        def get_thumbnail_dataset_by_id(*args):
            return DatasetData(name=f'dataset', id='dataset_id', last_modified_at=0,
                               type='thumbnails', size_in_bytes=-1, n_samples=-1, created_at=-1)
        self.api_workflow_client.datasets_api.get_dataset_by_id = get_thumbnail_dataset_by_id
        with self.assertRaises(ValueError):
            self.api_workflow_client.download_dataset('path/to/dir')

