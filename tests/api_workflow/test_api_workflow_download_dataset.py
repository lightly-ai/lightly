import shutil
from unittest import mock

import numpy as np
import PIL

import lightly
from lightly.api import api_workflow_download_dataset, download
from lightly.openapi_generated.swagger_client import DatasetData, DatasetEmbeddingData
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup


class TestApiWorkflowDownloadDataset(MockedApiWorkflowSetup):
    def setUp(self) -> None:
        MockedApiWorkflowSetup.setUp(self, dataset_id="dataset_0_id")
        self.api_workflow_client._tags_api.no_tags = 3

    def test_download_non_existing_tag(self):
        with self.assertRaises(ValueError):
            self.api_workflow_client.download_dataset(
                "path/to/dir", tag_name="this_is_not_a_real_tag_name"
            )

    def test_download_thumbnails(self):
        def get_thumbnail_dataset_by_id(*args):
            return DatasetData(
                name=f"dataset",
                id="dataset_id",
                last_modified_at=0,
                type="thumbnails",
                size_in_bytes=-1,
                n_samples=-1,
                created_at=-1,
            )

        self.api_workflow_client._datasets_api.get_dataset_by_id = (
            get_thumbnail_dataset_by_id
        )
        with self.assertRaises(ValueError):
            self.api_workflow_client.download_dataset("path/to/dir")

    def test_download_dataset(self):
        def my_func(read_url):
            return PIL.Image.fromarray(np.zeros((32, 32))).convert("RGB")

        # mock_get_image_from_readurl.return_value = PIL.Image.fromarray(np.zeros((32, 32)))
        lightly.api.api_workflow_download_dataset._get_image_from_read_url = my_func
        self.api_workflow_client.download_dataset(
            "path-to-dir-remove-me", tag_name="initial-tag"
        )
        shutil.rmtree("path-to-dir-remove-me")

    def test_get_embedding_data_by_name(self) -> None:
        embedding_0 = DatasetEmbeddingData(
            id="0",
            name="embedding_0",
            created_at=0,
            is_processed=False,
        )
        embedding_1 = DatasetEmbeddingData(
            id="1",
            name="embedding_1",
            created_at=1,
            is_processed=False,
        )
        with mock.patch.object(
            self.api_workflow_client._embeddings_api,
            "get_embeddings_by_dataset_id",
            return_value=[embedding_0, embedding_1],
        ) as mock_get_embeddings_by_dataset_id:
            embedding = self.api_workflow_client.get_embedding_data_by_name(
                name="embedding_0"
            )
            mock_get_embeddings_by_dataset_id.assert_called_once_with(
                dataset_id="dataset_0_id",
            )
            assert embedding == embedding_0

    def test_get_embedding_data_by_name__no_embedding_with_name(self) -> None:
        embedding_0 = DatasetEmbeddingData(
            id="0",
            name="embedding_0",
            created_at=0,
            is_processed=False,
        )
        with mock.patch.object(
            self.api_workflow_client._embeddings_api,
            "get_embeddings_by_dataset_id",
            return_value=[embedding_0],
        ) as mock_get_embeddings_by_dataset_id, self.assertRaisesRegex(
            ValueError,
            "There are no embeddings with name 'other_embedding' for dataset with id 'dataset_0_id'.",
        ):
            self.api_workflow_client.get_embedding_data_by_name(name="other_embedding")
            mock_get_embeddings_by_dataset_id.assert_called_once_with(
                dataset_id="dataset_0_id",
            )

    def test_download_embeddings_csv_by_id(self) -> None:
        with mock.patch.object(
            self.api_workflow_client._embeddings_api,
            "get_embeddings_csv_read_url_by_id",
            return_value="read_url",
        ) as mock_get_embeddings_csv_read_url_by_id, mock.patch.object(
            download, "download_and_write_file"
        ) as mock_download:
            self.api_workflow_client.download_embeddings_csv_by_id(
                embedding_id="embedding_id",
                output_path="embeddings.csv",
            )
            mock_get_embeddings_csv_read_url_by_id.assert_called_once_with(
                dataset_id="dataset_0_id",
                embedding_id="embedding_id",
            )
            mock_download.assert_called_once_with(
                url="read_url",
                output_path="embeddings.csv",
            )

    def test_download_embeddings_csv(self) -> None:
        with mock.patch.object(
            self.api_workflow_client,
            "get_all_embedding_data",
            return_value=[
                DatasetEmbeddingData(
                    id="0",
                    name="default_20221209_10h45m49s",
                    created_at=0,
                    is_processed=False,
                )
            ],
        ) as mock_get_all_embedding_data, mock.patch.object(
            self.api_workflow_client,
            "download_embeddings_csv_by_id",
        ) as mock_download_embeddings_csv_by_id:
            self.api_workflow_client.download_embeddings_csv(
                output_path="embeddings.csv"
            )
            mock_get_all_embedding_data.assert_called_once()
            mock_download_embeddings_csv_by_id.assert_called_once_with(
                embedding_id="0",
                output_path="embeddings.csv",
            )

    def test_download_embeddings_csv__no_default_embedding(self) -> None:
        with mock.patch.object(
            self.api_workflow_client,
            "get_all_embedding_data",
            return_value=[],
        ) as mock_get_all_embedding_data, self.assertRaisesRegex(
            RuntimeError,
            "Could not find embeddings for dataset with id 'dataset_0_id'.",
        ):
            self.api_workflow_client.download_embeddings_csv(
                output_path="embeddings.csv"
            )
            mock_get_all_embedding_data.assert_called_once()


def test__get_latest_default_embedding_data__no_default_embedding() -> None:
    custom_embedding = DatasetEmbeddingData(
        id="0",
        name="custom-name",
        created_at=0,
        is_processed=False,
    )
    embedding = api_workflow_download_dataset._get_latest_default_embedding_data(
        embeddings=[custom_embedding]
    )
    assert embedding is None
