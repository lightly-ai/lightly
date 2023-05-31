from unittest.mock import MagicMock

from lightly.api import ApiWorkflowClient, api_workflow_download_dataset
from lightly.openapi_generated.swagger_client import (
    DatasetEmbeddingData,
    FileNameFormat,
    TagsApi,
)
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup


class TestApiWorkflowExport(MockedApiWorkflowSetup):
    def setUp(self) -> None:
        MockedApiWorkflowSetup.setUp(self, dataset_id="dataset_0_id")
        self.api_workflow_client._tags_api.no_tags = 3

    def test_export_label_box_data_rows_by_tag_id(self):
        rows = self.api_workflow_client.export_label_box_data_rows_by_tag_id(
            tag_id="some-tag-id"
        )
        assert rows == [
            {
                "external_id": "2008_007291_jpg.rf.2fca436925b52ea33cf897125a34a2fb.jpg",
                "image_url": "https://api.lightly.ai/v1/datasets/62383ab8f9cb290cd83ab5f9/samples/62383cb7e6a0f29e3f31e233/readurlRedirect?type=CENSORED",
            }
        ]

    def test_export_label_box_data_rows_by_tag_name(self):
        rows = self.api_workflow_client.export_label_box_data_rows_by_tag_name(
            tag_name="initial-tag"
        )
        assert rows == [
            {
                "external_id": "2008_007291_jpg.rf.2fca436925b52ea33cf897125a34a2fb.jpg",
                "image_url": "https://api.lightly.ai/v1/datasets/62383ab8f9cb290cd83ab5f9/samples/62383cb7e6a0f29e3f31e233/readurlRedirect?type=CENSORED",
            }
        ]

    def test_export_label_box_v4_data_rows_by_tag_name(self):
        rows = self.api_workflow_client.export_label_box_v4_data_rows_by_tag_name(
            tag_name="initial-tag"
        )
        assert rows == [
            {
                "row_data": "http://localhost:5000/v1/datasets/6401d4534d2ed9112da782f5/samples/6401e455a6045a7faa79b20a/readurlRedirect?type=full&publicToken=token",
                "global_key": "image.png",
                "media_type": "IMAGE",
            }
        ]

    def test_export_label_box_v4_data_rows_by_tag_id(self):
        rows = self.api_workflow_client.export_label_box_v4_data_rows_by_tag_id(
            tag_id="some-tag-id"
        )
        assert rows == [
            {
                "row_data": "http://localhost:5000/v1/datasets/6401d4534d2ed9112da782f5/samples/6401e455a6045a7faa79b20a/readurlRedirect?type=full&publicToken=token",
                "global_key": "image.png",
                "media_type": "IMAGE",
            }
        ]

    def test_export_label_studio_tasks_by_tag_name(self):
        tasks = self.api_workflow_client.export_label_studio_tasks_by_tag_name(
            "initial-tag"
        )
        self.assertIsNotNone(tasks)
        self.assertTrue(all(isinstance(task, dict) for task in tasks))

    def test_export_tag_to_basic_filenames_and_read_urls(self):
        def mocked_export_tag_to_basic_filenames(
            dataset_id: str, tag_id: str, file_name_format: str
        ):
            return {
                FileNameFormat.NAME: "\n".join(["sample1.jpg", "sample2.jpg"]),
                FileNameFormat.REDIRECTED_READ_URL: "\n".join(
                    ["READ_URL_1", "READ_URL_2"]
                ),
                FileNameFormat.DATASOURCE_FULL: "\n".join(
                    ["s3://my_datasource/sample1.jpg", "s3://my_datasource/sample2.jpg"]
                ),
            }[file_name_format]

        mocked_client = MagicMock(spec=ApiWorkflowClient)
        mocked_client.dataset_id = "some_dataset_id"
        mocked_client._tags_api = MagicMock(spec_set=TagsApi)
        mocked_client._tags_api.export_tag_to_basic_filenames.side_effect = (
            mocked_export_tag_to_basic_filenames
        )

        data = ApiWorkflowClient.export_filenames_and_read_urls_by_tag_id(
            self=mocked_client, tag_id="tag_id"
        )

        assert data == [
            {
                "fileName": "sample1.jpg",
                "readUrl": "READ_URL_1",
                "datasourceUrl": "s3://my_datasource/sample1.jpg",
            },
            {
                "fileName": "sample2.jpg",
                "readUrl": "READ_URL_2",
                "datasourceUrl": "s3://my_datasource/sample2.jpg",
            },
        ]

    def test_export_filenames_by_tag_name(self):
        filenames = self.api_workflow_client.export_filenames_by_tag_name("initial-tag")
        self.assertIsNotNone(filenames)
        self.assertTrue(isinstance(filenames, str))


def test__get_latest_default_embedding_data() -> None:
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

    embedding = api_workflow_download_dataset._get_latest_default_embedding_data(
        embeddings=[embedding_0, embedding_1, embedding_2]
    )
    assert embedding == embedding_1


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
