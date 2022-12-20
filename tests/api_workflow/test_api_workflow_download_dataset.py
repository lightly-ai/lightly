import shutil
from unittest import mock

import numpy as np
import PIL

import lightly
from lightly.api import download
from lightly.openapi_generated.swagger_client import DatasetData, DatasetEmbeddingData
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup


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

    def test_download_embeddings_csv(self) -> None:
        with (
            mock.patch.object(
                self.api_workflow_client,
                "_get_last_default_embeddings_data",
                return_value=DatasetEmbeddingData(
                    id="0",
                    name="default_20221209_10h45m49s",
                    created_at=0,
                    is_processed=False,
                )
            ) as mock_get_last_default_embeddings_data,
            mock.patch.object(
                self.api_workflow_client._embeddings_api,
                "get_embeddings_csv_read_url_by_id",
                return_value="read-url",
            ) as mock_get_embeddings_csv_read_url_by_id,
            mock.patch.object(download, "download_and_write_file") as mock_download,
        ):
            self.api_workflow_client.download_embeddings_csv(output_path="embeddings.csv")
            mock_get_last_default_embeddings_data.assert_called_once()
            mock_get_embeddings_csv_read_url_by_id.assert_called_once_with(
                dataset_id="dataset_0_id",
                embedding_id="0",
            )
            mock_download.assert_called_once_with(
                url="read-url",
                output_path="embeddings.csv",
            )

    def test_download_embeddings_csv__no_default_embedding(self) -> None:
        with (
            mock.patch.object(
                self.api_workflow_client,
                "_get_last_default_embeddings_data",
                return_value=None,
            ) as mock_get_last_default_embeddings_data,
            self.assertRaisesRegex(
                RuntimeError,
                "Could not find embedding for dataset with id 'dataset_0_id'."
            )
        ):
            self.api_workflow_client.download_embeddings_csv(output_path="embeddings.csv")
            mock_get_last_default_embeddings_data.assert_called_once()

    def test__get_last_default_embeddings_data(self) -> None:
        embedding_0 = DatasetEmbeddingData(
            id="0",
            name="default_20221209_10h45m49s",
            created_at=0,
            is_processed=False,
        )
        embedding_1 = DatasetEmbeddingData(
            id="1",
            name="default_20221209_10h45m50s",
            created_at=1,
            is_processed=False,
        )
        embedding_2 = DatasetEmbeddingData(
            id="2",
            name="custom-name",
            created_at=2,
            is_processed=False,
        )
        with mock.patch.object(
            self.api_workflow_client._embeddings_api,
            "get_embeddings_by_dataset_id",
            return_value=[embedding_0, embedding_1, embedding_2],
        ) as mock_get_embeddings_by_dataset_id:
            embedding = self.api_workflow_client._get_last_default_embeddings_data()
            mock_get_embeddings_by_dataset_id.assert_called_once_with(
                dataset_id="dataset_0_id"
            )
            assert embedding == embedding_1

    def test__get_last_default_embeddings_data__no_default_embedding(self) -> None:
        custom_embedding = DatasetEmbeddingData(
            id="0",
            name="custom-name",
            created_at=0,
            is_processed=False,
        )
        with mock.patch.object(
            self.api_workflow_client._embeddings_api,
            "get_embeddings_by_dataset_id",
            return_value=[custom_embedding],
        ) as mock_get_embeddings_by_dataset_id:
            embedding = self.api_workflow_client._get_last_default_embeddings_data()
            mock_get_embeddings_by_dataset_id.assert_called_once_with(
                dataset_id="dataset_0_id"
            )
            assert embedding is None

    def test_export_label_box_data_rows_by_tag_name(self):
        rows = self.api_workflow_client.export_label_box_data_rows_by_tag_name('initial-tag')
        self.assertIsNotNone(rows)
        self.assertTrue(all(isinstance(row, dict) for row in rows))

    def test_export_label_studio_tasks_by_tag_name(self):
        tasks = self.api_workflow_client.export_label_studio_tasks_by_tag_name('initial-tag')
        self.assertIsNotNone(tasks)
        self.assertTrue(all(isinstance(task, dict) for task in tasks))

    def test_export_tag_to_basic_filenames_and_read_urls(self):
        filenames_and_read_urls = self.api_workflow_client.export_filenames_and_read_urls_by_tag_name('initial-tag')
        self.assertIsNotNone(filenames_and_read_urls)
        self.assertTrue(all(isinstance(filenames_and_read_url, dict) for filenames_and_read_url in filenames_and_read_urls))

    def test_export_filenames_by_tag_name(self):
        filenames = self.api_workflow_client.export_filenames_by_tag_name('initial-tag')
        self.assertIsNotNone(filenames)
        self.assertTrue(isinstance(filenames, str))