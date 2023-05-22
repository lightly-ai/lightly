from typing import List, Optional

import pytest
from pytest_mock import MockerFixture

from lightly.api import ApiWorkflowClient
from lightly.api.api_workflow_tags import TagDoesNotExistError
from lightly.openapi_client.models import TagCreator, TagData
from tests.api_workflow.utils import generate_id


def _get_tags(
    dataset_id: str, tag_name: str = "just-a-tag", prev_tag_id: Optional[str] = None
) -> List[TagData]:
    return [
        TagData(
            id=generate_id(),
            dataset_id=dataset_id,
            prev_tag_id=prev_tag_id,
            bit_mask_data="0x5",
            name=tag_name,
            tot_size=4,
            created_at=1577836800,
            changes=[],
        )
    ]


def test_create_tag_from_filenames(mocker: MockerFixture) -> None:
    dataset_id = generate_id()
    tags = _get_tags(dataset_id=dataset_id, tag_name="initial-tag")
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(ApiWorkflowClient, "get_all_tags", return_value=tags)
    mocked_get_filenames = mocker.patch.object(
        ApiWorkflowClient, "get_filenames", return_value=[f"file{i}" for i in range(3)]
    )
    mocked_api = mocker.MagicMock()

    client = ApiWorkflowClient()
    client._tags_api = mocked_api
    client._dataset_id = dataset_id
    client._creator = TagCreator.UNKNOWN
    client.create_tag_from_filenames(fnames_new_tag=["file2"], new_tag_name="some-tag")
    mocked_get_filenames.assert_called_once()
    mocked_api.create_tag_by_dataset_id.assert_called_once()
    kwargs = mocked_api.create_tag_by_dataset_id.call_args[1]
    # initial-tag is used as prev_tag_id when parent_tag_id is not given
    assert kwargs["tag_create_request"].prev_tag_id == tags[0].id
    assert kwargs["tag_create_request"].bit_mask_data == "0x4"


def test_create_tag_from_filenames__tag_exists(mocker: MockerFixture) -> None:
    tag_name = "some-tag"
    tags = _get_tags(dataset_id=generate_id(), tag_name=tag_name)
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(ApiWorkflowClient, "get_all_tags", return_value=tags)

    client = ApiWorkflowClient()

    with pytest.raises(RuntimeError) as exception:
        client.create_tag_from_filenames(fnames_new_tag=["file"], new_tag_name=tag_name)
        assert (
            str(exception.value) == "There already exists a tag with tag_name some-tag"
        )


def test_create_tag_from_filenames__no_tags(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(ApiWorkflowClient, "get_all_tags", return_value=[])

    client = ApiWorkflowClient()

    with pytest.raises(RuntimeError) as exception:
        client.create_tag_from_filenames(
            fnames_new_tag=["file"], new_tag_name="some-tag"
        )
        assert str(exception.value) == "There exists no initial-tag for this dataset."


def test_create_tag_from_filenames__file_not_found(mocker: MockerFixture) -> None:
    tags = _get_tags(dataset_id=generate_id(), tag_name="initial-tag")
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(ApiWorkflowClient, "get_all_tags", return_value=tags)
    mocked_get_filenames = mocker.patch.object(
        ApiWorkflowClient, "get_filenames", return_value=[f"file{i}" for i in range(3)]
    )

    client = ApiWorkflowClient()
    with pytest.raises(RuntimeError) as exception:
        client.create_tag_from_filenames(
            fnames_new_tag=["some-file"], new_tag_name="some-tag"
        )
        assert str(exception.value) == (
            "An error occured when creating the new subset! "
            "Out of the 1 filenames you provided "
            "to create a new tag, only 0 have been found on the server. "
            "Make sure you use the correct filenames. "
            "Valid filename example from the dataset: file0"
        )
    mocked_get_filenames.assert_called_once()


def test_get_filenames_in_tag(mocker: MockerFixture) -> None:
    tag = _get_tags(dataset_id=generate_id())[0]
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_get_filenames = mocker.patch.object(
        ApiWorkflowClient, "get_filenames", return_value=[f"file{i}" for i in range(3)]
    )

    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    result = client.get_filenames_in_tag(tag_data=tag)
    assert result == ["file0", "file2"]
    mocked_get_filenames.assert_called_once()


def test_get_filenames_in_tag__filenames_given(mocker: MockerFixture) -> None:
    tag = _get_tags(dataset_id=generate_id())[0]
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_get_filenames = mocker.patch.object(ApiWorkflowClient, "get_filenames")

    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    result = client.get_filenames_in_tag(
        tag_data=tag, filenames_on_server=[f"new-file-{i}" for i in range(3)]
    )
    assert result == ["new-file-0", "new-file-2"]
    mocked_get_filenames.assert_not_called()


def test_get_filenames_in_tag__exclude_parent_tag(mocker: MockerFixture) -> None:
    prev_tag_id = generate_id()
    dataset_id = generate_id()
    tag = _get_tags(dataset_id=dataset_id, prev_tag_id=prev_tag_id)[0]
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_get_filenames = mocker.patch.object(
        ApiWorkflowClient, "get_filenames", return_value=[f"file{i}" for i in range(3)]
    )
    mocked_response = mocker.MagicMock()
    mocked_response.bit_mask_data = "0x2"
    mocked_tag_arithmetics = mocker.MagicMock(return_value=mocked_response)
    mocked_api = mocker.MagicMock()
    mocked_api.perform_tag_arithmetics_bitmask = mocked_tag_arithmetics

    client = ApiWorkflowClient()
    client._dataset_id = dataset_id
    client._tags_api = mocked_api
    result = client.get_filenames_in_tag(tag_data=tag, exclude_parent_tag=True)
    assert result == ["file1"]
    mocked_get_filenames.assert_called_once()
    mocked_tag_arithmetics.assert_called_once()
    kwargs = mocked_tag_arithmetics.call_args[1]
    assert kwargs["dataset_id"] == dataset_id
    assert kwargs["tag_arithmetics_request"].tag_id2 == prev_tag_id


def test_get_all_tags(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_api = mocker.MagicMock()

    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    client._tags_api = mocked_api
    client.get_all_tags()
    mocked_api.get_tags_by_dataset_id.assert_called_once_with("dataset-id")


def test_get_tag_by_id(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_api = mocker.MagicMock()

    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    client._tags_api = mocked_api
    client.get_tag_by_id("tag-id")
    mocked_api.get_tag_by_tag_id.assert_called_once_with(
        dataset_id="dataset-id", tag_id="tag-id"
    )


def test_get_tag_name(mocker: MockerFixture) -> None:
    tag_name = "some-tag"
    tags = _get_tags(dataset_id=generate_id(), tag_name=tag_name)
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(ApiWorkflowClient, "get_all_tags", return_value=tags)
    mocked_get_tag = mocker.patch.object(ApiWorkflowClient, "get_tag_by_id")

    client = ApiWorkflowClient()
    client.get_tag_by_name(tag_name=tag_name)
    mocked_get_tag.assert_called_once_with(tags[0].id)


def test_get_tag_name__nonexisting(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(ApiWorkflowClient, "get_all_tags", return_value=[])

    client = ApiWorkflowClient()

    with pytest.raises(TagDoesNotExistError) as exception:
        client.get_tag_by_name(tag_name="some-tag")
        assert str(exception.value) == "Your tag_name does not exist: some-tag"


def test_delete_tag_by_id(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_api = mocker.MagicMock()

    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    client._tags_api = mocked_api
    client.delete_tag_by_id("tag-id")
    mocked_api.delete_tag_by_tag_id.assert_called_once_with(
        dataset_id="dataset-id", tag_id="tag-id"
    )
