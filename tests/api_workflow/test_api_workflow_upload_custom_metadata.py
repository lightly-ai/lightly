from pytest_mock import MockerFixture

from lightly.api import ApiWorkflowClient, api_workflow_upload_metadata
from lightly.openapi_generated.swagger_client.models import (
    SampleDataModes,
    SamplePartialMode,
    SampleUpdateRequest,
)
from lightly.utils.io import COCO_ANNOTATION_KEYS
from tests.api_workflow.utils import generate_id


def test_index_custom_metadata_by_filename(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    custom_metadata = {}
    custom_metadata[COCO_ANNOTATION_KEYS.images] = [
        {
            COCO_ANNOTATION_KEYS.images_filename: "file0",
            COCO_ANNOTATION_KEYS.images_id: "image-id0",
        },
        {
            COCO_ANNOTATION_KEYS.images_filename: "file1",
            COCO_ANNOTATION_KEYS.images_id: "image-id1",
        },
    ]
    custom_metadata[COCO_ANNOTATION_KEYS.custom_metadata] = [
        {COCO_ANNOTATION_KEYS.custom_metadata_image_id: "image-id2"},
        {COCO_ANNOTATION_KEYS.custom_metadata_image_id: "image-id0"},
    ]

    client = ApiWorkflowClient()
    result = client.index_custom_metadata_by_filename(custom_metadata=custom_metadata)
    assert result == {
        "file0": {COCO_ANNOTATION_KEYS.custom_metadata_image_id: "image-id0"},
        "file1": None,
    }


def test_upload_custom_metadata(mocker: MockerFixture) -> None:
    mocker.patch("tqdm.tqdm")
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    # retry should be called twice: once for get_samples_partial_by_dataset_id
    # and once for update_sample_by_id. get_samples_partial_by_dataset_id returns
    # only one valid sample file `file1`
    mocked_paginate_endpoint = mocker.patch.object(
        api_workflow_upload_metadata,
        "paginate_endpoint",
        side_effect=[
            [SampleDataModes(id=generate_id(), file_name="file1")],
            None,
        ],
    )
    mocked_retry = mocker.patch.object(
        api_workflow_upload_metadata,
        "retry",
        side_effect=[
            [SampleDataModes(id=generate_id(), file_name="file1")],
            None,
        ],
    )
    mocked_print_warning = mocker.patch.object(
        api_workflow_upload_metadata, "print_as_warning"
    )
    mocked_executor = mocker.patch.object(
        api_workflow_upload_metadata, "ThreadPoolExecutor"
    )
    mocked_executor.return_value.__enter__.return_value.map = (
        lambda fn, iterables, **_: map(fn, iterables)
    )
    mocked_samples_api = mocker.MagicMock()

    custom_metadata = {}
    custom_metadata[COCO_ANNOTATION_KEYS.images] = [
        {
            COCO_ANNOTATION_KEYS.images_filename: "file0",
            COCO_ANNOTATION_KEYS.images_id: "image-id0",
        },
        {
            COCO_ANNOTATION_KEYS.images_filename: "file1",
            COCO_ANNOTATION_KEYS.images_id: "image-id1",
        },
    ]
    custom_metadata[COCO_ANNOTATION_KEYS.custom_metadata] = [
        {COCO_ANNOTATION_KEYS.custom_metadata_image_id: "image-id2"},
        {COCO_ANNOTATION_KEYS.custom_metadata_image_id: "image-id1"},
        {COCO_ANNOTATION_KEYS.custom_metadata_image_id: "image-id0"},
    ]
    client = ApiWorkflowClient()
    client._dataset_id = "dataset-id"
    client._samples_api = mocked_samples_api
    client.upload_custom_metadata(custom_metadata=custom_metadata)

    # Only `file1` is a valid sample
    assert mocked_print_warning.call_count == 2
    warning_text = [
        call_args[0][0] for call_args in mocked_print_warning.call_args_list
    ]
    assert warning_text == [
        (
            "No image found for custom metadata annotation with image_id image-id2. "
            "This custom metadata annotation is skipped. "
        ),
        (
            "You tried to upload custom metadata for a sample with filename {file0}, "
            "but a sample with this filename does not exist on the server. "
            "This custom metadata annotation is skipped. "
        ),
    ]

    mocked_paginate_endpoint.assert_called_once_with(
        mocked_samples_api.get_samples_partial_by_dataset_id,
        dataset_id="dataset-id",
        mode=SamplePartialMode.FILENAMES,
        page_size=25000,
    )
    # First call: get_samples_partial_by_dataset_id
    args_first_call = mocked_paginate_endpoint.call_args_list[0][0]
    assert (
        # Check first positional argument
        args_first_call[0]
        == mocked_samples_api.get_samples_partial_by_dataset_id
    )
    # Second call: update_sample_by_id with the only valid sample
    mocked_retry.assert_called_once_with(
        mocked_samples_api.update_sample_by_id,
        sample_update_request=SampleUpdateRequest(
            custom_meta_data={
                COCO_ANNOTATION_KEYS.custom_metadata_image_id: "image-id1"
            }
        ),
    )
