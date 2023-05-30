import pytest
from pytest_mock import MockerFixture

from lightly.api import ApiWorkflowClient, api_workflow_download_dataset
from lightly.openapi_generated.swagger_client.models import (
    DatasetData,
    DatasetEmbeddingData,
    DatasetType,
    ImageType,
    TagData,
)
from tests.api_workflow.utils import generate_id


def test_download_dataset__no_image(mocker: MockerFixture) -> None:
    dataset_id = generate_id()
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_api = mocker.MagicMock()
    mocked_get_dataset_by_id = mocker.MagicMock(
        return_value=DatasetData(
            name="dataset",
            id=dataset_id,
            user_id=generate_id(),
            last_modified_at=0,
            type=DatasetType.IMAGES,
            img_type=ImageType.META,
            size_in_bytes=-1,
            n_samples=-1,
            created_at=0,
        )
    )
    mocked_api.get_dataset_by_id = mocked_get_dataset_by_id
    client = ApiWorkflowClient()
    client._dataset_id = dataset_id
    client._datasets_api = mocked_api
    with pytest.raises(ValueError) as exception:
        client.download_dataset(output_dir="path/to/dir")
        assert (
            str(exception.value)
            == f"Dataset with id {dataset_id} has no downloadable images!"
        )


def test_download_dataset__tag_missing(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_api = mocker.MagicMock()
    mocked_get_dataset_by_id = mocker.MagicMock(
        return_value=DatasetData(
            name="dataset",
            id=generate_id(),
            user_id=generate_id(),
            last_modified_at=0,
            type=DatasetType.IMAGES,
            img_type=ImageType.FULL,
            size_in_bytes=-1,
            n_samples=-1,
            created_at=0,
        )
    )
    mocked_api.get_dataset_by_id = mocked_get_dataset_by_id
    mocker.patch.object(ApiWorkflowClient, "get_all_tags", return_value=[])
    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    client._datasets_api = mocked_api
    with pytest.raises(ValueError) as exception:
        client.download_dataset(output_dir="path/to/dir", tag_name="some-tag")
        assert str(exception.value) == "Dataset with id dataset-id has no tag some-tag!"


def test_download_dataset__ok(mocker: MockerFixture) -> None:
    dataset_id = generate_id()

    mocked_get_dataset_by_id = mocker.MagicMock(
        return_value=DatasetData(
            name="dataset",
            id=dataset_id,
            user_id=generate_id(),
            last_modified_at=0,
            type=DatasetType.IMAGES,
            img_type=ImageType.FULL,
            size_in_bytes=-1,
            n_samples=-1,
            created_at=0,
        )
    )
    mocked_datasets_api = mocker.MagicMock()
    mocked_datasets_api.get_dataset_by_id = mocked_get_dataset_by_id

    mocked_get_sample_mappings_by_dataset_id = mocker.MagicMock(return_value=[1])
    mocked_mappings_api = mocker.MagicMock()
    mocked_mappings_api.get_sample_mappings_by_dataset_id = (
        mocked_get_sample_mappings_by_dataset_id
    )

    mocked_get_sample_image_read_url_by_id = mocker.MagicMock(
        side_effect=RuntimeError("some error")
    )
    mocked_samples_api = mocker.MagicMock()
    mocked_samples_api.get_sample_image_read_url_by_id = (
        mocked_get_sample_image_read_url_by_id
    )

    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(
        ApiWorkflowClient,
        "get_all_tags",
        return_value=[
            TagData(
                id=generate_id(),
                dataset_id=dataset_id,
                prev_tag_id=None,
                bit_mask_data="0x1",
                name="some-tag",
                tot_size=4,
                created_at=1577836800,
                changes=[],
            )
        ],
    )
    mocker.patch.object(
        ApiWorkflowClient, "get_filenames", return_value=[f"file{i}" for i in range(3)]
    )
    mocker.patch.object(api_workflow_download_dataset, "_get_image_from_read_url")
    mocker.patch.object(api_workflow_download_dataset, "_make_dir_and_save_image")
    mocked_warning = mocker.patch("warnings.warn")
    mocker.patch("tqdm.tqdm")
    mocked_executor = mocker.patch.object(
        api_workflow_download_dataset, "ThreadPoolExecutor"
    )
    mocked_executor.return_value.__enter__.return_value.map = (
        lambda fn, iterables, **_: map(fn, iterables)
    )

    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    client._datasets_api = mocked_datasets_api
    client._mappings_api = mocked_mappings_api
    client._samples_api = mocked_samples_api

    client.download_dataset(output_dir="path/to/dir", tag_name="some-tag")

    assert mocked_warning.call_count == 2
    warning_text = [str(call_args[0][0]) for call_args in mocked_warning.call_args_list]
    assert warning_text == [
        "Downloading of image file0 failed with error some error",
        "Warning: Unsuccessful download! Failed at image: 0",
    ]


def test_get_embedding_data_by_name(mocker: MockerFixture) -> None:
    embedding_0 = DatasetEmbeddingData(
        id=generate_id(),
        name="embedding_0",
        created_at=0,
        is_processed=False,
    )
    embedding_1 = DatasetEmbeddingData(
        id=generate_id(),
        name="embedding_1",
        created_at=1,
        is_processed=False,
    )
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(
        ApiWorkflowClient,
        "get_all_embedding_data",
        return_value=[embedding_0, embedding_1],
    )
    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"

    embedding = client.get_embedding_data_by_name(name="embedding_0")
    assert embedding == embedding_0


def test_get_embedding_data_by_name__no_embedding_with_name(
    mocker: MockerFixture,
) -> None:
    embedding = DatasetEmbeddingData(
        id=generate_id(),
        name="embedding",
        created_at=0,
        is_processed=False,
    )
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(
        ApiWorkflowClient, "get_all_embedding_data", return_value=[embedding]
    )
    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    with pytest.raises(ValueError) as exception:
        client.get_embedding_data_by_name(name="other_embedding")
        assert str(exception.value) == (
            "There are no embeddings with name 'other_embedding' "
            "for dataset with id 'dataset-id'."
        )


def test_download_embeddings_csv_by_id(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_download = mocker.patch.object(
        api_workflow_download_dataset.download, "download_and_write_file"
    )
    mocked_api = mocker.MagicMock()
    mocked_get_embeddings_csv_read_url_by_id = mocker.MagicMock(return_value="read_url")
    mocked_api.get_embeddings_csv_read_url_by_id = (
        mocked_get_embeddings_csv_read_url_by_id
    )
    mocker.patch.object(
        api_workflow_download_dataset,
        "_get_latest_default_embedding_data",
        return_value=None,
    )
    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    client._embeddings_api = mocked_api

    client.download_embeddings_csv_by_id(
        embedding_id="embedding_id",
        output_path="embeddings.csv",
    )
    mocked_get_embeddings_csv_read_url_by_id.assert_called_once_with(
        dataset_id="dataset-id",
        embedding_id="embedding_id",
    )
    mocked_download.assert_called_once_with(
        url="read_url",
        output_path="embeddings.csv",
    )


def test_download_embeddings_csv(mocker: MockerFixture) -> None:
    embedding_id = generate_id()

    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mock_get_all_embedding_data = mocker.patch.object(
        api_workflow_download_dataset,
        "_get_latest_default_embedding_data",
        return_value=DatasetEmbeddingData(
            id=embedding_id,
            name="default_20221209_10h45m49s",
            created_at=0,
            is_processed=False,
        ),
    )
    mocker.patch.object(ApiWorkflowClient, "get_all_embedding_data")
    mock_download_embeddings_csv_by_id = mocker.patch.object(
        ApiWorkflowClient,
        "download_embeddings_csv_by_id",
    )

    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    client.download_embeddings_csv(output_path="embeddings.csv")
    mock_get_all_embedding_data.assert_called_once()
    mock_download_embeddings_csv_by_id.assert_called_once_with(
        embedding_id=embedding_id,
        output_path="embeddings.csv",
    )


def test_download_embeddings_csv__no_default_embedding(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_get_all_embedding_data = mocker.patch.object(
        ApiWorkflowClient, "get_all_embedding_data", return_value=[]
    )
    mocker.patch.object(
        api_workflow_download_dataset,
        "_get_latest_default_embedding_data",
        return_value=None,
    )
    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    with pytest.raises(RuntimeError) as exception:
        client.download_embeddings_csv(output_path="embeddings.csv")
    assert (
        str(exception.value)
        == "Could not find embeddings for dataset with id 'dataset-id'."
    )
    mocked_get_all_embedding_data.assert_called_once()


def test__get_latest_default_embedding_data__no_default_embedding() -> None:
    custom_embedding = DatasetEmbeddingData(
        id=generate_id(),
        name="custom-name",
        created_at=0,
        is_processed=False,
    )
    embedding = api_workflow_download_dataset._get_latest_default_embedding_data(
        embeddings=[custom_embedding]
    )
    assert embedding is None
