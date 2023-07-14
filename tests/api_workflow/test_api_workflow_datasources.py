import pytest
from pytest_mock import MockerFixture

from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client.models import (
    DatasourceConfigAzure,
    DatasourceConfigGCS,
    DatasourceConfigLOCAL,
    DatasourceConfigS3,
    DatasourceConfigS3DelegatedAccess,
    DatasourceRawSamplesDataRow,
)
from lightly.openapi_generated.swagger_client.models.datasource_config_verify_data import (
    DatasourceConfigVerifyData,
)
from lightly.openapi_generated.swagger_client.models.datasource_config_verify_data_errors import (
    DatasourceConfigVerifyDataErrors,
)


def test__download_raw_files(mocker: MockerFixture) -> None:
    mock_response_1 = mocker.MagicMock()
    mock_response_1.has_more = True
    mock_response_1.data = [
        DatasourceRawSamplesDataRow(file_name="/file1", read_url="url1"),
        DatasourceRawSamplesDataRow(file_name="file2", read_url="url2"),
    ]

    mock_response_2 = mocker.MagicMock()
    mock_response_2.has_more = False
    mock_response_2.data = [
        DatasourceRawSamplesDataRow(file_name="./file3", read_url="url3"),
        DatasourceRawSamplesDataRow(file_name="file2", read_url="url2"),
    ]

    mocked_method = mocker.MagicMock(side_effect=[mock_response_1, mock_response_2])
    mocked_pbar = mocker.MagicMock()
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_warning = mocker.patch("warnings.warn")
    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    result = client._download_raw_files(
        download_function=mocked_method,
        progress_bar=mocked_pbar,
    )
    kwargs = mocked_method.call_args[1]
    assert "relevant_filenames_file_name" not in kwargs
    assert mocked_pbar.update.call_count == 2
    assert mocked_warning.call_count == 3
    warning_text = [str(call_args[0][0]) for call_args in mocked_warning.call_args_list]
    assert warning_text == [
        (
            "Absolute file paths like /file1 are not supported"
            " in relevant filenames file None due to blob storage"
        ),
        (
            "Using dot notation ('./', '../') like in ./file3 is not supported"
            " in relevant filenames file None due to blob storage"
        ),
        ("Duplicate filename file2 in relevant filenames file None"),
    ]
    assert len(result) == 1
    assert result[0][0] == "file2"


def test_get_prediction_read_url(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    client._datasources_api = mocked_api
    client.get_prediction_read_url("test.json")
    mocked_method = (
        mocked_api.get_prediction_file_read_url_from_datasource_by_dataset_id
    )
    mocked_method.assert_called_once_with(
        dataset_id="dataset-id", file_name="test.json"
    )


def test_download_new_raw_samples(mocker: MockerFixture) -> None:
    from_timestamp = 2
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(
        ApiWorkflowClient, "get_processed_until_timestamp", return_value=from_timestamp
    )
    current_time = 5
    mocker.patch("time.time", return_value=current_time)
    mocked_download = mocker.patch.object(ApiWorkflowClient, "download_raw_samples")
    mocked_update_timestamp = mocker.patch.object(
        ApiWorkflowClient, "update_processed_until_timestamp"
    )
    client = ApiWorkflowClient()
    client.download_new_raw_samples()
    mocked_download.assert_called_once_with(
        from_=from_timestamp + 1,
        to=current_time,
        relevant_filenames_file_name=None,
        use_redirected_read_url=False,
    )
    mocked_update_timestamp.assert_called_once_with(timestamp=current_time)


def test_download_new_raw_samples__from_beginning(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(
        ApiWorkflowClient, "get_processed_until_timestamp", return_value=0
    )
    current_time = 5
    mocker.patch("time.time", return_value=current_time)
    mocked_download = mocker.patch.object(ApiWorkflowClient, "download_raw_samples")
    mocked_update_timestamp = mocker.patch.object(
        ApiWorkflowClient, "update_processed_until_timestamp"
    )
    client = ApiWorkflowClient()
    client.download_new_raw_samples()
    mocked_download.assert_called_once_with(
        from_=0,
        to=current_time,
        relevant_filenames_file_name=None,
        use_redirected_read_url=False,
    )
    mocked_update_timestamp.assert_called_once_with(timestamp=current_time)


def test_download_raw_samples_predictions__relevant_filenames_artifact_id(
    mocker: MockerFixture,
) -> None:
    mock_response = mocker.MagicMock()
    mock_response.has_more = False
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_api = mocker.MagicMock()
    mocked_method = mocker.MagicMock(return_value=mock_response)
    mocked_api.get_list_of_raw_samples_predictions_from_datasource_by_dataset_id = (
        mocked_method
    )
    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    client._datasources_api = mocked_api
    client.download_raw_predictions(
        task_name="task", run_id="foo", relevant_filenames_artifact_id="bar"
    )
    kwargs = mocked_method.call_args[1]
    assert kwargs.get("relevant_filenames_run_id") == "foo"
    assert kwargs.get("relevant_filenames_artifact_id") == "bar"

    # should raise ValueError when only run_id is given
    with pytest.raises(ValueError):
        client.download_raw_predictions(task_name="foobar", run_id="foo")
    # should raise ValueError when only relevant_filenames_artifact_id is given
    with pytest.raises(ValueError):
        client.download_raw_predictions(
            task_name="foobar", relevant_filenames_artifact_id="bar"
        )


def test_download_raw_samples_metadata__relevant_filenames_artifact_id(
    mocker: MockerFixture,
) -> None:
    mock_response = mocker.MagicMock()
    mock_response.has_more = False
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_api = mocker.MagicMock()
    mocked_method = mocker.MagicMock(return_value=mock_response)
    mocked_api.get_list_of_raw_samples_metadata_from_datasource_by_dataset_id = (
        mocked_method
    )
    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    client._datasources_api = mocked_api
    client.download_raw_metadata(run_id="foo", relevant_filenames_artifact_id="bar")
    kwargs = mocked_method.call_args[1]
    assert kwargs.get("relevant_filenames_run_id") == "foo"
    assert kwargs.get("relevant_filenames_artifact_id") == "bar"

    # should raise ValueError when only run_id is given
    with pytest.raises(ValueError):
        client.download_raw_metadata(run_id="foo")
    # should raise ValueError when only relevant_filenames_artifact_id is given
    with pytest.raises(ValueError):
        client.download_raw_metadata(relevant_filenames_artifact_id="bar")


def test_get_processed_until_timestamp(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_datasources_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    client._datasources_api = mocked_datasources_api
    client.get_processed_until_timestamp()
    mocked_method = (
        mocked_datasources_api.get_datasource_processed_until_timestamp_by_dataset_id
    )
    mocked_method.assert_called_once_with(dataset_id="dataset-id")


def test_set_azure_config(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_datasources_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._datasources_api = mocked_datasources_api
    client._dataset_id = "dataset-id"
    client.set_azure_config(
        container_name="my-container/name",
        account_name="my-account-name",
        sas_token="my-sas-token",
        thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
    )
    kwargs = mocked_datasources_api.update_datasource_by_dataset_id.call_args[1]
    assert isinstance(
        kwargs["datasource_config"].actual_instance, DatasourceConfigAzure
    )


def test_set_gcs_config(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_datasources_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._datasources_api = mocked_datasources_api
    client._dataset_id = "dataset-id"
    client.set_gcs_config(
        resource_path="gs://my-bucket/my-dataset",
        project_id="my-project-id",
        credentials="my-credentials",
        thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
    )
    kwargs = mocked_datasources_api.update_datasource_by_dataset_id.call_args[1]
    assert isinstance(kwargs["datasource_config"].actual_instance, DatasourceConfigGCS)


def test_set_local_config(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_datasources_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._datasources_api = mocked_datasources_api
    client._dataset_id = "dataset-id"
    client.set_local_config(
        resource_path="http://localhost:1234/path/to/my/data",
        thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
    )
    kwargs = mocked_datasources_api.update_datasource_by_dataset_id.call_args[1]
    assert isinstance(
        kwargs["datasource_config"].actual_instance, DatasourceConfigLOCAL
    )


def test_set_s3_config(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_datasources_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._datasources_api = mocked_datasources_api
    client._dataset_id = "dataset-id"
    client.set_s3_config(
        resource_path="s3://my-bucket/my-dataset",
        thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
        region="eu-central-1",
        access_key="my-access-key",
        secret_access_key="my-secret-access-key",
    )
    kwargs = mocked_datasources_api.update_datasource_by_dataset_id.call_args[1]
    assert isinstance(kwargs["datasource_config"].actual_instance, DatasourceConfigS3)


def test_set_s3_delegated_access_config(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_datasources_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._datasources_api = mocked_datasources_api
    client._dataset_id = "dataset-id"
    client.set_s3_delegated_access_config(
        resource_path="s3://my-bucket/my-dataset",
        thumbnail_suffix=".lightly/thumbnails/[filename]-thumb-[extension]",
        region="eu-central-1",
        role_arn="arn:aws:iam::000000000000:role.test",
        external_id="my-external-id",
    )
    kwargs = mocked_datasources_api.update_datasource_by_dataset_id.call_args[1]
    assert isinstance(
        kwargs["datasource_config"].actual_instance, DatasourceConfigS3DelegatedAccess
    )


def test_update_processed_until_timestamp(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_datasources_api = mocker.MagicMock()
    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    client._datasources_api = mocked_datasources_api
    client.update_processed_until_timestamp(10)
    kwargs = mocked_datasources_api.update_datasource_processed_until_timestamp_by_dataset_id.call_args[
        1
    ]
    assert kwargs["dataset_id"] == "dataset-id"
    assert (
        kwargs["datasource_processed_until_timestamp_request"].processed_until_timestamp
        == 10
    )


def test_list_datasource_permissions(mocker: MockerFixture) -> None:
    client = ApiWorkflowClient(token="abc")
    client._dataset_id = "dataset-id"
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


def test_list_datasource_permissions__error(mocker: MockerFixture) -> None:
    client = ApiWorkflowClient(token="abc")
    client._dataset_id = "dataset-id"
    client._datasources_api.verify_datasource_by_dataset_id = mocker.MagicMock(
        return_value=DatasourceConfigVerifyData(
            canRead=True,
            canWrite=True,
            canList=False,
            canOverwrite=True,
            errors=DatasourceConfigVerifyDataErrors(
                canRead=None, canWrite=None, canList="error message", canOverwrite=None
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
