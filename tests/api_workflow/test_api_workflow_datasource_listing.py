import pytest
import tqdm
from pytest_mock import MockerFixture

from lightly.api import ApiWorkflowClient, api_workflow_listing
from lightly.openapi_generated.swagger_client.models import DatasourceRawSamplesDataRow
from lightly.openapi_generated.swagger_client.models.datasource_processed_until_timestamp_response import (
    DatasourceProcessedUntilTimestampResponse,
)
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data import (
    DatasourceRawSamplesData,
)
from lightly.openapi_generated.swagger_client.rest import ApiException


class TestListingMixin:
    @pytest.mark.parametrize("with_retry", [True, False])
    def test_download_raw_samples(
        self, mocker: MockerFixture, with_retry: bool
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
        side_effects = [response]
        if with_retry:
            side_effects.insert(
                0, ApiException(status=500, reason="Internal Server Error")
            )
        mocker.patch.object(
            client._datasources_api,
            "get_list_of_raw_samples_from_datasource_by_dataset_id",
            side_effect=side_effects,
        )
        assert client.download_raw_samples() == [("file1", "url1"), ("file2", "url2")]

    @pytest.mark.parametrize("with_retry", [True, False])
    def test_download_raw_predictions(
        self, mocker: MockerFixture, with_retry: bool
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
        side_effects = [response]
        if with_retry:
            side_effects.insert(
                0, ApiException(status=500, reason="Internal Server Error")
            )
        mocker.patch.object(
            client._datasources_api,
            "get_list_of_raw_samples_predictions_from_datasource_by_dataset_id",
            side_effect=side_effects,
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

    @pytest.mark.parametrize("with_retry", [True, False])
    def test_download_raw_metadata(
        self, mocker: MockerFixture, with_retry: bool
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
        side_effects = [response]
        if with_retry:
            side_effects.insert(
                0, ApiException(status=500, reason="Internal Server Error")
            )
        mocker.patch.object(
            client._datasources_api,
            "get_list_of_raw_samples_metadata_from_datasource_by_dataset_id",
            side_effect=side_effects,
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
        # This test is not tested with retry, as it calls three functions that are called with retry:
        # - get_processed_until_timestamp
        # - download_raw_samples
        # - update_processed_until_timestamp
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

    @pytest.mark.parametrize("with_retry", [True, False])
    def test_get_processed_until_timestamp(
        self, mocker: MockerFixture, with_retry: bool
    ) -> None:
        response = DatasourceProcessedUntilTimestampResponse(processedUntilTimestamp=5)
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        side_effects = [response]
        if with_retry:
            side_effects.insert(
                0, ApiException(status=500, reason="Internal Server Error")
            )
        mocker.patch.object(
            client._datasources_api,
            "get_datasource_processed_until_timestamp_by_dataset_id",
            side_effect=side_effects,
        )
        assert client.get_processed_until_timestamp() == 5
        client._datasources_api.get_datasource_processed_until_timestamp_by_dataset_id.assert_called_with(
            dataset_id="dataset-id"
        )

    @pytest.mark.parametrize("with_retry", [True, False])
    def test_update_processed_until_timestamp(
        self, mocker: MockerFixture, with_retry: bool
    ) -> None:
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        side_effects = [None]
        if with_retry:
            side_effects.insert(
                0, ApiException(status=500, reason="Internal Server Error")
            )
        mocker.patch.object(
            client._datasources_api,
            "update_datasource_processed_until_timestamp_by_dataset_id",
            side_effect=side_effects,
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

    @pytest.mark.parametrize("with_retry", [True, False])
    def test__download_raw_files_iter(
        self, mocker: MockerFixture, with_retry: bool
    ) -> None:
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
        if with_retry:
            side_effects = [
                ApiException(status=500, reason="Internal Server Error"),
                response_1,
                response_2,
            ]
        else:
            side_effects = [response_1, response_2]
        download_function = mocker.MagicMock(side_effect=side_effects)
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
        expected_calls = [
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
        if with_retry:
            # Assert that only the first call is retried.
            expected_calls.insert(0, expected_calls[0])
        download_function.assert_has_calls(expected_calls)
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
        assert not api_workflow_listing._sample_unseen_and_valid(
            sample=DatasourceRawSamplesDataRow(fileName="/file1", readUrl="url1"),
            relevant_filenames_file_name=None,
            listed_filenames=set(),
        )

    with pytest.warns(UserWarning, match="Using dot notation"):
        assert not api_workflow_listing._sample_unseen_and_valid(
            sample=DatasourceRawSamplesDataRow(fileName="./file1", readUrl="url1"),
            relevant_filenames_file_name=None,
            listed_filenames=set(),
        )

    with pytest.warns(UserWarning, match="Duplicate filename file1"):
        assert not api_workflow_listing._sample_unseen_and_valid(
            sample=DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
            relevant_filenames_file_name=None,
            listed_filenames={"file1"},
        )

    assert api_workflow_listing._sample_unseen_and_valid(
        sample=DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
        relevant_filenames_file_name=None,
        listed_filenames=set(),
    )
