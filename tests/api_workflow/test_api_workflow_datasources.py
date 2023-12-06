import pytest
import tqdm
from pytest_mock import MockerFixture

from lightly.api import ApiWorkflowClient, api_workflow_datasources
from lightly.openapi_generated.swagger_client.models import (
    DatasourceConfigAzure,
    DatasourceConfigGCS,
    DatasourceConfigLOCAL,
    DatasourceConfigS3,
    DatasourceConfigS3DelegatedAccess,
    DatasourcePurpose,
    DatasourceRawSamplesDataRow,
)
from lightly.openapi_generated.swagger_client.models.datasource_config_verify_data import (
    DatasourceConfigVerifyData,
)
from lightly.openapi_generated.swagger_client.models.datasource_config_verify_data_errors import (
    DatasourceConfigVerifyDataErrors,
)
from lightly.openapi_generated.swagger_client.models.datasource_processed_until_timestamp_response import (
    DatasourceProcessedUntilTimestampResponse,
)
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data import (
    DatasourceRawSamplesData,
)


class TestDatasourcesMixin:
    def test_download_raw_samples(self, mocker: MockerFixture) -> None:
        response = DatasourceRawSamplesData(
            hasMore=False,
            cursor="",
            data=[
                DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
                DatasourceRawSamplesDataRow(fileName="file2", readUrl="url2"),
            ],
        )
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "get_list_of_raw_samples_from_datasource_by_dataset_id",
            side_effect=[response],
        )
        assert client.download_raw_samples() == [("file1", "url1"), ("file2", "url2")]

    def test_download_raw_predictions(self, mocker: MockerFixture) -> None:
        response = DatasourceRawSamplesData(
            hasMore=False,
            cursor="",
            data=[
                DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
                DatasourceRawSamplesDataRow(fileName="file2", readUrl="url2"),
            ],
        )
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "get_list_of_raw_samples_predictions_from_datasource_by_dataset_id",
            side_effect=[response],
        )
        assert client.download_raw_predictions(task_name="task") == [
            ("file1", "url1"),
            ("file2", "url2"),
        ]

    def test_download_raw_predictions_iter(self, mocker: MockerFixture) -> None:
        response_1 = DatasourceRawSamplesData(
            hasMore=True,
            cursor="cursor1",
            data=[
                DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
                DatasourceRawSamplesDataRow(fileName="file2", readUrl="url2"),
            ],
        )
        response_2 = DatasourceRawSamplesData(
            hasMore=False,
            cursor="cursor2",
            data=[
                DatasourceRawSamplesDataRow(fileName="file3", readUrl="url3"),
                DatasourceRawSamplesDataRow(fileName="file4", readUrl="url4"),
            ],
        )
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "get_list_of_raw_samples_predictions_from_datasource_by_dataset_id",
            side_effect=[response_1, response_2],
        )
        assert list(client.download_raw_predictions_iter(task_name="task")) == [
            ("file1", "url1"),
            ("file2", "url2"),
            ("file3", "url3"),
            ("file4", "url4"),
        ]
        client._datasources_api.get_list_of_raw_samples_predictions_from_datasource_by_dataset_id.assert_has_calls(
            [
                mocker.call(
                    dataset_id="dataset-id",
                    task_name="task",
                    var_from=0,
                    to=mocker.ANY,
                    use_redirected_read_url=False,
                ),
                mocker.call(
                    dataset_id="dataset-id",
                    task_name="task",
                    cursor="cursor1",
                    use_redirected_read_url=False,
                ),
            ]
        )

    def test_download_raw_predictions_iter__relevant_filenames_artifact_id(
        self,
        mocker: MockerFixture,
    ) -> None:
        response = DatasourceRawSamplesData(
            hasMore=False,
            cursor="",
            data=[
                DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
                DatasourceRawSamplesDataRow(fileName="file2", readUrl="url2"),
            ],
        )
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "get_list_of_raw_samples_predictions_from_datasource_by_dataset_id",
            side_effect=[response],
        )
        assert list(
            client.download_raw_predictions_iter(
                task_name="task",
                run_id="run-id",
                relevant_filenames_artifact_id="relevant-filenames",
            )
        ) == [
            ("file1", "url1"),
            ("file2", "url2"),
        ]
        client._datasources_api.get_list_of_raw_samples_predictions_from_datasource_by_dataset_id.assert_called_once_with(
            dataset_id="dataset-id",
            task_name="task",
            var_from=0,
            to=mocker.ANY,
            relevant_filenames_run_id="run-id",
            relevant_filenames_artifact_id="relevant-filenames",
            use_redirected_read_url=False,
        )

        # should raise ValueError when only run_id is given
        with pytest.raises(ValueError):
            next(
                client.download_raw_predictions_iter(task_name="task", run_id="run-id")
            )

        # should raise ValueError when only relevant_filenames_artifact_id is given
        with pytest.raises(ValueError):
            next(
                client.download_raw_predictions_iter(
                    task_name="task",
                    relevant_filenames_artifact_id="relevant-filenames",
                )
            )

    def test_download_raw_metadata(self, mocker: MockerFixture) -> None:
        response = DatasourceRawSamplesData(
            hasMore=False,
            cursor="",
            data=[
                DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
                DatasourceRawSamplesDataRow(fileName="file2", readUrl="url2"),
            ],
        )
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "get_list_of_raw_samples_metadata_from_datasource_by_dataset_id",
            side_effect=[response],
        )
        assert client.download_raw_metadata() == [
            ("file1", "url1"),
            ("file2", "url2"),
        ]

    def test_download_raw_metadata_iter(self, mocker: MockerFixture) -> None:
        response_1 = DatasourceRawSamplesData(
            hasMore=True,
            cursor="cursor1",
            data=[
                DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
                DatasourceRawSamplesDataRow(fileName="file2", readUrl="url2"),
            ],
        )
        response_2 = DatasourceRawSamplesData(
            hasMore=False,
            cursor="cursor2",
            data=[
                DatasourceRawSamplesDataRow(fileName="file3", readUrl="url3"),
                DatasourceRawSamplesDataRow(fileName="file4", readUrl="url4"),
            ],
        )
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "get_list_of_raw_samples_metadata_from_datasource_by_dataset_id",
            side_effect=[response_1, response_2],
        )
        assert list(client.download_raw_metadata_iter()) == [
            ("file1", "url1"),
            ("file2", "url2"),
            ("file3", "url3"),
            ("file4", "url4"),
        ]
        client._datasources_api.get_list_of_raw_samples_metadata_from_datasource_by_dataset_id.assert_has_calls(
            [
                mocker.call(
                    dataset_id="dataset-id",
                    var_from=0,
                    to=mocker.ANY,
                    use_redirected_read_url=False,
                ),
                mocker.call(
                    dataset_id="dataset-id",
                    cursor="cursor1",
                    use_redirected_read_url=False,
                ),
            ]
        )

    def test_download_raw_metadata_iter__relevant_filenames_artifact_id(
        self, mocker: MockerFixture
    ) -> None:
        response = DatasourceRawSamplesData(
            hasMore=False,
            cursor="",
            data=[
                DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
                DatasourceRawSamplesDataRow(fileName="file2", readUrl="url2"),
            ],
        )
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "get_list_of_raw_samples_metadata_from_datasource_by_dataset_id",
            side_effect=[response],
        )
        assert list(
            client.download_raw_metadata_iter(
                run_id="run-id",
                relevant_filenames_artifact_id="relevant-filenames",
            )
        ) == [
            ("file1", "url1"),
            ("file2", "url2"),
        ]
        client._datasources_api.get_list_of_raw_samples_metadata_from_datasource_by_dataset_id.assert_called_once_with(
            dataset_id="dataset-id",
            var_from=0,
            to=mocker.ANY,
            relevant_filenames_run_id="run-id",
            relevant_filenames_artifact_id="relevant-filenames",
            use_redirected_read_url=False,
        )

        # should raise ValueError when only run_id is given
        with pytest.raises(ValueError):
            next(client.download_raw_metadata_iter(run_id="run-id"))

        # should raise ValueError when only relevant_filenames_artifact_id is given
        with pytest.raises(ValueError):
            next(
                client.download_raw_metadata_iter(
                    relevant_filenames_artifact_id="relevant-filenames",
                )
            )

    def test_download_new_raw_samples(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        client.get_processed_until_timestamp = mocker.MagicMock(return_value=2)
        mocker.patch("time.time", return_value=5)
        mocker.patch.object(client, "download_raw_samples")
        mocker.patch.object(client, "update_processed_until_timestamp")
        client.download_new_raw_samples()
        client.download_raw_samples.assert_called_once_with(
            from_=2 + 1,
            to=5,
            relevant_filenames_file_name=None,
            use_redirected_read_url=False,
        )
        client.update_processed_until_timestamp.assert_called_once_with(timestamp=5)

    def test_download_new_raw_samples__from_beginning(
        self, mocker: MockerFixture
    ) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        client.get_processed_until_timestamp = mocker.MagicMock(return_value=2)
        mocker.patch("time.time", return_value=5)
        mocker.patch.object(client, "download_raw_samples")
        mocker.patch.object(client, "update_processed_until_timestamp")
        client.download_new_raw_samples()
        client.download_raw_samples.assert_called_once_with(
            from_=3,
            to=5,
            relevant_filenames_file_name=None,
            use_redirected_read_url=False,
        )
        client.update_processed_until_timestamp.assert_called_once_with(timestamp=5)

    def test_get_processed_until_timestamp(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "get_datasource_processed_until_timestamp_by_dataset_id",
            return_value=DatasourceProcessedUntilTimestampResponse(
                processedUntilTimestamp=5
            ),
        )
        assert client.get_processed_until_timestamp() == 5
        client._datasources_api.get_datasource_processed_until_timestamp_by_dataset_id.assert_called_once_with(
            dataset_id="dataset-id"
        )

    def test_update_processed_until_timestamp(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "update_datasource_processed_until_timestamp_by_dataset_id",
        )
        client.update_processed_until_timestamp(timestamp=10)
        kwargs = client._datasources_api.update_datasource_processed_until_timestamp_by_dataset_id.call_args[
            1
        ]
        assert kwargs["dataset_id"] == "dataset-id"
        assert (
            kwargs[
                "datasource_processed_until_timestamp_request"
            ].processed_until_timestamp
            == 10
        )

    def test_set_azure_config(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "update_datasource_by_dataset_id",
        )
        client.set_azure_config(
            container_name="my-container/name",
            account_name="my-account-name",
            sas_token="my-sas-token",
            thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
        )
        kwargs = client._datasources_api.update_datasource_by_dataset_id.call_args[1]
        assert isinstance(
            kwargs["datasource_config"].actual_instance, DatasourceConfigAzure
        )

    def test_set_gcs_config(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "update_datasource_by_dataset_id",
        )
        client.set_gcs_config(
            resource_path="gs://my-bucket/my-dataset",
            project_id="my-project-id",
            credentials="my-credentials",
            thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
        )
        kwargs = client._datasources_api.update_datasource_by_dataset_id.call_args[1]
        assert isinstance(
            kwargs["datasource_config"].actual_instance, DatasourceConfigGCS
        )

    def test_set_local_config(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "update_datasource_by_dataset_id",
        )
        client.set_local_config(
            web_server_location="http://localhost:1234",
            relative_path="path/to/my/data",
            thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
            purpose=DatasourcePurpose.INPUT,
        )
        kwargs = client._datasources_api.update_datasource_by_dataset_id.call_args[1]
        datasource_config = kwargs["datasource_config"].actual_instance
        assert isinstance(datasource_config, DatasourceConfigLOCAL)
        assert datasource_config.type == "LOCAL"
        assert datasource_config.web_server_location == "http://localhost:1234"
        assert datasource_config.full_path == "path/to/my/data"
        assert (
            datasource_config.thumb_suffix
            == ".lightly/thumbnails/[filename]-thumb-[extension]"
        )
        assert datasource_config.purpose == DatasourcePurpose.INPUT

        # Test defaults
        client.set_local_config()
        kwargs = client._datasources_api.update_datasource_by_dataset_id.call_args[1]
        datasource_config = kwargs["datasource_config"].actual_instance
        assert isinstance(datasource_config, DatasourceConfigLOCAL)
        assert datasource_config.type == "LOCAL"
        assert datasource_config.web_server_location == "http://localhost:3456"
        assert datasource_config.full_path == ""
        assert (
            datasource_config.thumb_suffix
            == ".lightly/thumbnails/[filename]_thumb.[extension]"
        )
        assert datasource_config.purpose == DatasourcePurpose.INPUT_OUTPUT

    def test_set_s3_config(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "update_datasource_by_dataset_id",
        )
        client.set_s3_config(
            resource_path="s3://my-bucket/my-dataset",
            thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
            region="eu-central-1",
            access_key="my-access-key",
            secret_access_key="my-secret-access-key",
        )
        kwargs = client._datasources_api.update_datasource_by_dataset_id.call_args[1]
        assert isinstance(
            kwargs["datasource_config"].actual_instance, DatasourceConfigS3
        )

    def test_set_s3_delegated_access_config(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "update_datasource_by_dataset_id",
        )
        client.set_s3_delegated_access_config(
            resource_path="s3://my-bucket/my-dataset",
            thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
            region="eu-central-1",
            role_arn="arn:aws:iam::000000000000:role.test",
            external_id="my-external-id",
        )
        kwargs = client._datasources_api.update_datasource_by_dataset_id.call_args[1]
        assert isinstance(
            kwargs["datasource_config"].actual_instance,
            DatasourceConfigS3DelegatedAccess,
        )

    def test_get_prediction_read_url(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "get_prediction_file_read_url_from_datasource_by_dataset_id",
            return_value="read-url",
        )
        assert client.get_prediction_read_url(filename="test.json") == "read-url"
        client._datasources_api.get_prediction_file_read_url_from_datasource_by_dataset_id.assert_called_once_with(
            dataset_id="dataset-id", file_name="test.json"
        )

    def test_get_custom_embedding_read_url(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mocker.patch.object(
            client._datasources_api,
            "get_custom_embedding_file_read_url_from_datasource_by_dataset_id",
            return_value="read-url",
        )
        assert (
            client.get_custom_embedding_read_url(filename="embeddings.csv")
            == "read-url"
        )
        client._datasources_api.get_custom_embedding_file_read_url_from_datasource_by_dataset_id.assert_called_once_with(
            dataset_id="dataset-id", file_name="embeddings.csv"
        )

    def test_list_datasource_permissions(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        client._datasources_api.verify_datasource_by_dataset_id = mocker.MagicMock(
            return_value=DatasourceConfigVerifyData(
                canRead=True,
                canWrite=True,
                canList=False,
                canOverwrite=True,
                errors=None,
            ),
        )
        assert client.list_datasource_permissions() == {
            "can_read": True,
            "can_write": True,
            "can_list": False,
            "can_overwrite": True,
        }

    def test_list_datasource_permissions__error(self, mocker: MockerFixture) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        client._datasources_api.verify_datasource_by_dataset_id = mocker.MagicMock(
            return_value=DatasourceConfigVerifyData(
                canRead=True,
                canWrite=True,
                canList=False,
                canOverwrite=True,
                errors=DatasourceConfigVerifyDataErrors(
                    canRead=None,
                    canWrite=None,
                    canList="error message",
                    canOverwrite=None,
                ),
            ),
        )
        assert client.list_datasource_permissions() == {
            "can_read": True,
            "can_write": True,
            "can_list": False,
            "can_overwrite": True,
            "errors": {
                "can_list": "error message",
            },
        }

    def test__download_raw_files(self, mocker: MockerFixture) -> None:
        response = DatasourceRawSamplesData(
            hasMore=False,
            cursor="",
            data=[
                DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
                DatasourceRawSamplesDataRow(fileName="file2", readUrl="url2"),
            ],
        )
        download_function = mocker.MagicMock(side_effect=[response])
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        assert client._download_raw_files(
            download_function=download_function,
        ) == [("file1", "url1"), ("file2", "url2")]

    def test__download_raw_files_iter(self, mocker: MockerFixture) -> None:
        response_1 = DatasourceRawSamplesData(
            hasMore=True,
            cursor="cursor1",
            data=[
                DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
                DatasourceRawSamplesDataRow(fileName="file2", readUrl="url2"),
            ],
        )
        response_2 = DatasourceRawSamplesData(
            hasMore=False,
            cursor="cursor2",
            data=[
                DatasourceRawSamplesDataRow(fileName="file3", readUrl="url3"),
                DatasourceRawSamplesDataRow(fileName="file4", readUrl="url4"),
            ],
        )
        download_function = mocker.MagicMock(side_effect=[response_1, response_2])
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        progress_bar = mocker.spy(tqdm, "tqdm")
        assert list(
            client._download_raw_files_iter(
                download_function=download_function,
                from_=0,
                to=5,
                relevant_filenames_file_name="relevant-filenames",
                use_redirected_read_url=True,
                progress_bar=progress_bar,
                foo="bar",
            )
        ) == [
            ("file1", "url1"),
            ("file2", "url2"),
            ("file3", "url3"),
            ("file4", "url4"),
        ]
        download_function.assert_has_calls(
            [
                mocker.call(
                    dataset_id="dataset-id",
                    var_from=0,
                    to=5,
                    relevant_filenames_file_name="relevant-filenames",
                    use_redirected_read_url=True,
                    foo="bar",
                ),
                mocker.call(
                    dataset_id="dataset-id",
                    cursor="cursor1",
                    relevant_filenames_file_name="relevant-filenames",
                    use_redirected_read_url=True,
                    foo="bar",
                ),
            ]
        )
        assert progress_bar.update.call_count == 4

    def test__download_raw_files_iter__no_relevant_filenames(
        self, mocker: MockerFixture
    ) -> None:
        response = DatasourceRawSamplesData(hasMore=False, cursor="", data=[])
        download_function = mocker.MagicMock(side_effect=[response])
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        list(client._download_raw_files_iter(download_function=download_function))
        assert "relevant_filenames_file_name" not in download_function.call_args[1]

    def test__download_raw_files_iter__warning(self, mocker: MockerFixture) -> None:
        response = DatasourceRawSamplesData(
            hasMore=False,
            cursor="",
            data=[
                DatasourceRawSamplesDataRow(fileName="/file1", readUrl="url1"),
            ],
        )
        download_function = mocker.MagicMock(side_effect=[response])
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        with pytest.warns(UserWarning, match="Absolute file paths like /file1"):
            list(client._download_raw_files_iter(download_function=download_function))


def test__sample_unseen_and_valid() -> None:
    with pytest.warns(UserWarning, match="Absolute file paths like /file1"):
        assert not api_workflow_datasources._sample_unseen_and_valid(
            sample=DatasourceRawSamplesDataRow(fileName="/file1", readUrl="url1"),
            relevant_filenames_file_name=None,
            listed_filenames=set(),
        )

    with pytest.warns(UserWarning, match="Using dot notation"):
        assert not api_workflow_datasources._sample_unseen_and_valid(
            sample=DatasourceRawSamplesDataRow(fileName="./file1", readUrl="url1"),
            relevant_filenames_file_name=None,
            listed_filenames=set(),
        )

    with pytest.warns(UserWarning, match="Duplicate filename file1"):
        assert not api_workflow_datasources._sample_unseen_and_valid(
            sample=DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
            relevant_filenames_file_name=None,
            listed_filenames={"file1"},
        )

    assert api_workflow_datasources._sample_unseen_and_valid(
        sample=DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
        relevant_filenames_file_name=None,
        listed_filenames=set(),
    )
