import pytest
import tqdm
from pytest_mock import MockerFixture

from lightly.api import ApiWorkflowClient, api_workflow_datasource_listing
from lightly.openapi_generated.swagger_client.models import DatasourceRawSamplesDataRow
from lightly.openapi_generated.swagger_client.models.datasource_processed_until_timestamp_response import (
    DatasourceProcessedUntilTimestampResponse,
)
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data import (
    DatasourceRawSamplesData,
)
from lightly.openapi_generated.swagger_client.models.divide_and_conquer_cursor_data import (
    DivideAndConquerCursorData,
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
        dnc_response = DivideAndConquerCursorData(
            cursors=["divide_and_conquer_cursor1"]
        )
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        side_effects = [response]
        if with_retry:
            side_effects.insert(
                0, ApiException(status=500, reason="Internal Server Error")
            )

        mock_download_function = mocker.patch.object(
            client._datasources_api,
            "get_list_of_raw_samples_from_datasource_by_dataset_id",
            side_effect=side_effects,
        )
        mock_dnc_function = mocker.patch.object(
            client._datasources_api,
            "get_divide_and_conquer_list_of_raw_samples_from_datasource_by_dataset_id",
            return_value=dnc_response,
        )

        result = client.download_raw_samples()

        assert result == [("file1", "url1"), ("file2", "url2")]

        # Verify divide and conquer function was called correctly
        mock_dnc_function.assert_called_once_with(
            dataset_id="dataset-id", var_from=0, to=mocker.ANY, dnc_shards=1
        )

        # Verify download function was called with the cursor from divide and conquer
        mock_download_function.assert_called_with(
            dataset_id="dataset-id",
            cursor="divide_and_conquer_cursor1",
            use_redirected_read_url=False,
        )

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
        dnc_response = DivideAndConquerCursorData(
            cursors=["divide_and_conquer_cursor1"]
        )
        mocker.patch.object(
            client._datasources_api,
            "get_divide_and_conquer_list_of_raw_samples_predictions_from_datasource_by_dataset_id",
            return_value=dnc_response,
        )
        assert client.download_raw_predictions(task_name="task") == [
            ("file1", "url1"),
            ("file2", "url2"),
        ]

    def test_download_raw_predictions_iter(self, mocker: MockerFixture) -> None:
        response_1 = DatasourceRawSamplesData(
            hasMore=True,
            cursor="continuous_cursor1",
            data=[
                DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
                DatasourceRawSamplesDataRow(fileName="file2", readUrl="url2"),
            ],
        )
        response_2 = DatasourceRawSamplesData(
            hasMore=False,
            cursor="continuous_cursor2",
            data=[
                DatasourceRawSamplesDataRow(fileName="file3", readUrl="url3"),
                DatasourceRawSamplesDataRow(fileName="file4", readUrl="url4"),
            ],
        )
        dnc_response = DivideAndConquerCursorData(
            cursors=["divide_and_conquer_cursor1", "divide_and_conquer_cursor2"]
        )
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")

        mock_download_function = mocker.patch.object(
            client._datasources_api,
            "get_list_of_raw_samples_predictions_from_datasource_by_dataset_id",
            side_effect=[response_1, response_2],
        )
        mock_dnc_function = mocker.patch.object(
            client._datasources_api,
            "get_divide_and_conquer_list_of_raw_samples_predictions_from_datasource_by_dataset_id",
            return_value=dnc_response,
        )

        result = list(client.download_raw_predictions_iter(task_name="task"))

        assert result == [
            ("file1", "url1"),
            ("file2", "url2"),
            ("file3", "url3"),
            ("file4", "url4"),
        ]

        # Verify divide and conquer function was called correctly
        mock_dnc_function.assert_called_once_with(
            dataset_id="dataset-id",
            var_from=0,
            to=mocker.ANY,
            dnc_shards=1,
            task_name="task",
        )

        # Verify download function was called with both divide and conquer cursors
        # and then with continuous cursors from responses
        mock_download_function.assert_has_calls(
            [
                mocker.call(
                    dataset_id="dataset-id",
                    cursor="divide_and_conquer_cursor1",
                    task_name="task",
                    use_redirected_read_url=False,
                ),
                mocker.call(
                    dataset_id="dataset-id",
                    cursor="continuous_cursor1",
                    task_name="task",
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
        dnc_response = DivideAndConquerCursorData(
            cursors=["divide_and_conquer_cursor1"]
        )
        mocker.patch.object(
            client._datasources_api,
            "get_divide_and_conquer_list_of_raw_samples_predictions_from_datasource_by_dataset_id",
            return_value=dnc_response,
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
        client._datasources_api.get_divide_and_conquer_list_of_raw_samples_predictions_from_datasource_by_dataset_id.assert_called_once_with(
            dataset_id="dataset-id",
            var_from=0,
            to=mocker.ANY,
            dnc_shards=1,
            task_name="task",
            relevant_filenames_run_id="run-id",
            relevant_filenames_artifact_id="relevant-filenames",
        )
        client._datasources_api.get_list_of_raw_samples_predictions_from_datasource_by_dataset_id.assert_called_once_with(
            dataset_id="dataset-id",
            task_name="task",
            cursor="divide_and_conquer_cursor1",
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
        dnc_response = DivideAndConquerCursorData(
            cursors=["divide_and_conquer_cursor1"]
        )
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        side_effects = [response]
        if with_retry:
            side_effects.insert(
                0, ApiException(status=500, reason="Internal Server Error")
            )

        mock_download_function = mocker.patch.object(
            client._datasources_api,
            "get_list_of_raw_samples_metadata_from_datasource_by_dataset_id",
            side_effect=side_effects,
        )
        mock_dnc_function = mocker.patch.object(
            client._datasources_api,
            "get_divide_and_conquer_list_of_raw_samples_metadata_from_datasource_by_dataset_id",
            return_value=dnc_response,
        )

        result = client.download_raw_metadata()

        assert result == [("file1", "url1"), ("file2", "url2")]

        # Verify divide and conquer function was called correctly
        mock_dnc_function.assert_called_once_with(
            dataset_id="dataset-id", var_from=0, to=mocker.ANY, dnc_shards=1
        )

        # Verify download function was called with the cursor from divide and conquer
        mock_download_function.assert_called_with(
            dataset_id="dataset-id",
            cursor="divide_and_conquer_cursor1",
            use_redirected_read_url=False,
        )

    def test_download_raw_metadata_iter(self, mocker: MockerFixture) -> None:
        response_1 = DatasourceRawSamplesData(
            hasMore=True,
            cursor="continuous_cursor1",
            data=[
                DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
                DatasourceRawSamplesDataRow(fileName="file2", readUrl="url2"),
            ],
        )
        response_2 = DatasourceRawSamplesData(
            hasMore=False,
            cursor="continuous_cursor2",
            data=[
                DatasourceRawSamplesDataRow(fileName="file3", readUrl="url3"),
                DatasourceRawSamplesDataRow(fileName="file4", readUrl="url4"),
            ],
        )
        dnc_response = DivideAndConquerCursorData(
            cursors=["divide_and_conquer_cursor1"]
        )
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")

        mock_download_function = mocker.patch.object(
            client._datasources_api,
            "get_list_of_raw_samples_metadata_from_datasource_by_dataset_id",
            side_effect=[response_1, response_2],
        )
        mock_dnc_function = mocker.patch.object(
            client._datasources_api,
            "get_divide_and_conquer_list_of_raw_samples_metadata_from_datasource_by_dataset_id",
            return_value=dnc_response,
        )

        result = list(client.download_raw_metadata_iter())

        assert result == [
            ("file1", "url1"),
            ("file2", "url2"),
            ("file3", "url3"),
            ("file4", "url4"),
        ]

        # Verify divide and conquer function was called correctly
        mock_dnc_function.assert_called_once_with(
            dataset_id="dataset-id", var_from=0, to=mocker.ANY, dnc_shards=1
        )

        # Verify download function was called with divide and conquer cursor
        # and then with continuous cursor from response
        mock_download_function.assert_has_calls(
            [
                mocker.call(
                    dataset_id="dataset-id",
                    cursor="divide_and_conquer_cursor1",
                    use_redirected_read_url=False,
                ),
                mocker.call(
                    dataset_id="dataset-id",
                    cursor="continuous_cursor1",
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
        dnc_response = DivideAndConquerCursorData(
            cursors=["divide_and_conquer_cursor1"]
        )
        mocker.patch.object(
            client._datasources_api,
            "get_divide_and_conquer_list_of_raw_samples_metadata_from_datasource_by_dataset_id",
            return_value=dnc_response,
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
        client._datasources_api.get_divide_and_conquer_list_of_raw_samples_metadata_from_datasource_by_dataset_id.assert_called_once_with(
            dataset_id="dataset-id",
            var_from=0,
            to=mocker.ANY,
            dnc_shards=1,
            relevant_filenames_run_id="run-id",
            relevant_filenames_artifact_id="relevant-filenames",
        )
        client._datasources_api.get_list_of_raw_samples_metadata_from_datasource_by_dataset_id.assert_called_once_with(
            dataset_id="dataset-id",
            cursor="divide_and_conquer_cursor1",
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
            divide_and_conquer_shards=1,
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
            divide_and_conquer_shards=1,
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
        dnc_response = DivideAndConquerCursorData(
            cursors=["divide_and_conquer_cursor1"]
        )
        download_function = mocker.MagicMock(side_effect=[response])
        dnc_function = mocker.MagicMock(return_value=dnc_response)
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        assert client._download_raw_files(
            download_function=download_function,
            dnc_function=dnc_function,
        ) == [("file1", "url1"), ("file2", "url2")]

    @pytest.mark.parametrize("with_retry", [True, False])
    def test__download_raw_files_divide_and_conquer_iter(
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
        dnc_response = DivideAndConquerCursorData(
            cursors=["divide_and_conquer_cursor1"]
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
        dnc_function = mocker.MagicMock(return_value=dnc_response)
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        mock_progress_bar = mocker.MagicMock()
        assert list(
            client._download_raw_files_divide_and_conquer_iter(
                download_function=download_function,
                dnc_function=dnc_function,
                from_=0,
                to=5,
                relevant_filenames_file_name="relevant-filenames",
                use_redirected_read_url=True,
                progress_bar=mock_progress_bar,
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
                cursor="divide_and_conquer_cursor1",
                relevant_filenames_file_name="relevant-filenames",
                use_redirected_read_url=True,
            ),
            mocker.call(
                dataset_id="dataset-id",
                cursor="cursor1",
                relevant_filenames_file_name="relevant-filenames",
                use_redirected_read_url=True,
            ),
        ]
        if with_retry:
            # Assert that only the first call is retried.
            expected_calls.insert(0, expected_calls[0])
        download_function.assert_has_calls(expected_calls)
        assert mock_progress_bar.update.call_count == 4

    def test__download_raw_files_divide_and_conquer_iter__no_relevant_filenames(
        self, mocker: MockerFixture
    ) -> None:
        response = DatasourceRawSamplesData(hasMore=False, cursor="", data=[])
        dnc_response = DivideAndConquerCursorData(cursors=["dnc_cursor1"])
        download_function = mocker.MagicMock(side_effect=[response])
        dnc_function = mocker.MagicMock(return_value=dnc_response)
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        list(client._download_raw_files_divide_and_conquer_iter(download_function=download_function, dnc_function=dnc_function))
        assert "relevant_filenames_file_name" not in download_function.call_args[1]

    def test__download_raw_files_divide_and_conquer_iter__warning(self, mocker: MockerFixture) -> None:
        response = DatasourceRawSamplesData(
            hasMore=False,
            cursor="",
            data=[
                DatasourceRawSamplesDataRow(fileName="/file1", readUrl="url1"),
            ],
        )
        download_function = mocker.MagicMock(side_effect=[response])
        dnc_response = DivideAndConquerCursorData(cursors=["dnc_cursor1"])
        dnc_function = mocker.MagicMock(return_value=dnc_response)
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        with pytest.warns(UserWarning, match="Absolute file paths like /file1"):
            list(client._download_raw_files_divide_and_conquer_iter(download_function=download_function, dnc_function=dnc_function))

    def test__get_divide_and_conquer_list_cursors__basic(
        self, mocker: MockerFixture
    ) -> None:
        """Test basic functionality of _get_divide_and_conquer_list_cursors."""
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        dnc_function = mocker.MagicMock()
        dnc_response = DivideAndConquerCursorData(
            cursors=["divide_and_conquer_cursor1", "divide_and_conquer_cursor2"]
        )
        dnc_function.return_value = dnc_response

        cursors = client._get_divide_and_conquer_list_cursors(
            dnc_function=dnc_function, divide_and_conquer_shards=2
        )

        assert cursors == ["divide_and_conquer_cursor1", "divide_and_conquer_cursor2"]
        dnc_function.assert_called_once_with(
            dataset_id="dataset-id", var_from=0, to=mocker.ANY, dnc_shards=2
        )

    def test__get_divide_and_conquer_list_cursors__with_parameters(
        self, mocker: MockerFixture
    ) -> None:
        """Test _get_divide_and_conquer_list_cursors with various parameters."""
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        dnc_function = mocker.MagicMock()
        dnc_response = DivideAndConquerCursorData(
            cursors=["divide_and_conquer_cursor1"]
        )
        dnc_function.return_value = dnc_response

        cursors = client._get_divide_and_conquer_list_cursors(
            dnc_function=dnc_function,
            from_=0,
            to=100,
            relevant_filenames_file_name="relevant.txt",
            divide_and_conquer_shards=1,
            task_name="test_task",
        )

        assert cursors == ["divide_and_conquer_cursor1"]
        dnc_function.assert_called_once_with(
            dataset_id="dataset-id",
            var_from=0,
            to=100,
            dnc_shards=1,
            relevant_filenames_file_name="relevant.txt",
            task_name="test_task",
        )

    def test__get_divide_and_conquer_list_cursors__shards_minimum(
        self, mocker: MockerFixture
    ) -> None:
        """Test _get_divide_and_conquer_list_cursors ensures minimum 1 shard."""
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        dnc_function = mocker.MagicMock()
        dnc_response = DivideAndConquerCursorData(
            cursors=["divide_and_conquer_cursor1"]
        )
        dnc_function.return_value = dnc_response

        cursors = client._get_divide_and_conquer_list_cursors(
            dnc_function=dnc_function, divide_and_conquer_shards=0
        )

        assert cursors == ["divide_and_conquer_cursor1"]
        dnc_function.assert_called_once_with(
            dataset_id="dataset-id", var_from=0, to=mocker.ANY, dnc_shards=1
        )

    @pytest.mark.parametrize("with_retry", [True, False])
    def test__get_divide_and_conquer_list_cursors__with_retry(
        self, mocker: MockerFixture, with_retry: bool
    ) -> None:
        """Test _get_divide_and_conquer_list_cursors with retry functionality."""
        client = ApiWorkflowClient(token="abc", dataset_id="dataset-id")
        dnc_function = mocker.MagicMock()
        dnc_response = DivideAndConquerCursorData(
            cursors=["divide_and_conquer_cursor1"]
        )

        side_effects = [dnc_response]
        if with_retry:
            side_effects.insert(
                0, ApiException(status=500, reason="Internal Server Error")
            )
        dnc_function.side_effect = side_effects

        cursors = client._get_divide_and_conquer_list_cursors(
            dnc_function=dnc_function, divide_and_conquer_shards=1
        )

        assert cursors == ["divide_and_conquer_cursor1"]
        expected_call_count = 2 if with_retry else 1
        assert dnc_function.call_count == expected_call_count

        # Verify that both calls (if retry happened) used the same parameters
        for call in dnc_function.call_args_list:
            assert call[1]["dataset_id"] == "dataset-id"
            assert call[1]["var_from"] == 0
            assert call[1]["dnc_shards"] == 1


def test__sample_unseen_and_valid() -> None:
    with pytest.warns(UserWarning, match="Absolute file paths like /file1"):
        assert not api_workflow_datasource_listing._sample_unseen_and_valid(
            sample=DatasourceRawSamplesDataRow(fileName="/file1", readUrl="url1"),
            relevant_filenames_file_name=None,
            listed_filenames=set(),
        )

    with pytest.warns(UserWarning, match="Using dot notation"):
        assert not api_workflow_datasource_listing._sample_unseen_and_valid(
            sample=DatasourceRawSamplesDataRow(fileName="./file1", readUrl="url1"),
            relevant_filenames_file_name=None,
            listed_filenames=set(),
        )

    with pytest.warns(UserWarning, match="Duplicate filename file1"):
        assert not api_workflow_datasource_listing._sample_unseen_and_valid(
            sample=DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
            relevant_filenames_file_name=None,
            listed_filenames={"file1"},
        )

    assert api_workflow_datasource_listing._sample_unseen_and_valid(
        sample=DatasourceRawSamplesDataRow(fileName="file1", readUrl="url1"),
        relevant_filenames_file_name=None,
        listed_filenames=set(),
    )
