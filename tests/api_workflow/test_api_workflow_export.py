from pytest_mock import MockerFixture

from lightly.api import ApiWorkflowClient, api_workflow_export
from lightly.openapi_generated.swagger_client.models import FileNameFormat, TagData
from tests.api_workflow.utils import generate_id


def _get_tag(dataset_id: str, tag_name: str) -> TagData:
    return TagData(
        id=generate_id(),
        dataset_id=dataset_id,
        prev_tag_id=None,
        bit_mask_data="0x1",
        name=tag_name,
        tot_size=4,
        created_at=1577836800,
        changes=[],
    )


def test_export_filenames_by_tag_id(mocker: MockerFixture) -> None:
    dataset_id = generate_id()
    mocked_paginate = mocker.patch.object(
        api_workflow_export,
        "paginate_endpoint",
        side_effect=[iter(["file0\nfile1"])],
    )
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_api = mocker.MagicMock()

    client = ApiWorkflowClient()
    client._dataset_id = dataset_id
    client._tags_api = mocked_api
    data = client.export_filenames_by_tag_id(tag_id="tag_id")

    assert data == "file0\nfile1"
    mocked_paginate.assert_called_once_with(
        mocked_api.export_tag_to_basic_filenames,
        dataset_id=dataset_id,
        tag_id="tag_id",
    )


def test_export_filenames_by_tag_id__two_pages(mocker: MockerFixture) -> None:
    dataset_id = generate_id()
    mocked_paginate = mocker.patch.object(
        api_workflow_export,
        "paginate_endpoint",
        side_effect=[
            # Simulate two pages.
            iter(["file0\nfile1", "file2\nfile3"])
        ],
    )
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_api = mocker.MagicMock()

    client = ApiWorkflowClient()
    client._dataset_id = dataset_id
    client._tags_api = mocked_api
    data = client.export_filenames_by_tag_id(tag_id="tag_id")

    assert data == "file0\nfile1\nfile2\nfile3"
    mocked_paginate.assert_called_once_with(
        mocked_api.export_tag_to_basic_filenames,
        dataset_id=dataset_id,
        tag_id="tag_id",
    )


def test_export_filenames_and_read_urls_by_tag_id(mocker: MockerFixture) -> None:
    dataset_id = generate_id()
    mocked_paginate = mocker.patch.object(
        api_workflow_export,
        "paginate_endpoint",
        side_effect=[
            iter(["file0\nfile1"]),
            iter(["read_url0\nread_url1"]),
            iter(["datasource_url0\ndatasource_url1"]),
        ],
    )
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_api = mocker.MagicMock()

    client = ApiWorkflowClient()
    client._dataset_id = dataset_id
    client._tags_api = mocked_api
    data = client.export_filenames_and_read_urls_by_tag_id(tag_id="tag_id")

    assert data == [
        {
            "fileName": "file0",
            "readUrl": "read_url0",
            "datasourceUrl": "datasource_url0",
        },
        {
            "fileName": "file1",
            "readUrl": "read_url1",
            "datasourceUrl": "datasource_url1",
        },
    ]
    assert mocked_paginate.call_count == 3
    file_name_format_call_args = [
        call_args[1].get("file_name_format")
        for call_args in mocked_paginate.call_args_list
    ]
    assert file_name_format_call_args == [
        FileNameFormat.NAME,
        FileNameFormat.REDIRECTED_READ_URL,
        FileNameFormat.DATASOURCE_FULL,
    ]


def test_export_filenames_and_read_urls_by_tag_id__two_pages(
    mocker: MockerFixture,
) -> None:
    dataset_id = generate_id()
    mocked_paginate = mocker.patch.object(
        api_workflow_export,
        "paginate_endpoint",
        side_effect=[
            # Simulate two pages.
            iter(["file0\nfile1", "file2\nfile3"]),
            iter(["read_url0\nread_url1", "read_url2\nread_url3"]),
            iter(
                ["datasource_url0\ndatasource_url1", "datasource_url2\ndatasource_url3"]
            ),
        ],
    )
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_api = mocker.MagicMock()

    client = ApiWorkflowClient()
    client._dataset_id = dataset_id
    client._tags_api = mocked_api
    data = client.export_filenames_and_read_urls_by_tag_id(tag_id="tag_id")

    assert data == [
        {
            "fileName": "file0",
            "readUrl": "read_url0",
            "datasourceUrl": "datasource_url0",
        },
        {
            "fileName": "file1",
            "readUrl": "read_url1",
            "datasourceUrl": "datasource_url1",
        },
        {
            "fileName": "file2",
            "readUrl": "read_url2",
            "datasourceUrl": "datasource_url2",
        },
        {
            "fileName": "file3",
            "readUrl": "read_url3",
            "datasourceUrl": "datasource_url3",
        },
    ]
    assert mocked_paginate.call_count == 3
    file_name_format_call_args = [
        call_args[1].get("file_name_format")
        for call_args in mocked_paginate.call_args_list
    ]
    assert file_name_format_call_args == [
        FileNameFormat.NAME,
        FileNameFormat.REDIRECTED_READ_URL,
        FileNameFormat.DATASOURCE_FULL,
    ]


def test_export_filenames_by_tag_name(mocker: MockerFixture) -> None:
    dataset_id = generate_id()
    tag_name = "some-tag"
    tag = _get_tag(dataset_id=dataset_id, tag_name=tag_name)
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_get_tag = mocker.patch.object(
        ApiWorkflowClient, "get_tag_by_name", return_value=tag
    )
    mocked_export = mocker.patch.object(ApiWorkflowClient, "export_filenames_by_tag_id")
    client = ApiWorkflowClient()
    client._dataset_id = dataset_id
    client.export_filenames_by_tag_name(tag_name)
    mocked_get_tag.assert_called_once_with(tag_name)
    mocked_export.assert_called_once_with(tag.id)


def test_export_label_box_data_rows_by_tag_id(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_paginate = mocker.patch.object(api_workflow_export, "paginate_endpoint")
    mocked_api = mocker.MagicMock()
    mocked_warning = mocker.patch("warnings.warn")

    client = ApiWorkflowClient()
    client._dataset_id = generate_id()
    client._tags_api = mocked_api
    client.export_label_box_data_rows_by_tag_id(tag_id="tag_id")
    mocked_paginate.assert_called_once()
    call_args = mocked_paginate.call_args[0]
    assert call_args[0] == mocked_api.export_tag_to_label_box_data_rows
    warning_text = str(mocked_warning.call_args[0][0])
    assert warning_text == (
        "This method exports data in the deprecated Labelbox v3 format and "
        "will be removed in the future. Use export_label_box_v4_data_rows_by_tag_id "
        "to export data in the Labelbox v4 format instead."
    )


def test_export_label_box_data_rows_by_tag_name(mocker: MockerFixture) -> None:
    dataset_id = generate_id()
    tag_name = "some-tag"
    tag = _get_tag(dataset_id=dataset_id, tag_name=tag_name)
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_get_tag = mocker.patch.object(
        ApiWorkflowClient, "get_tag_by_name", return_value=tag
    )
    mocked_export = mocker.patch.object(
        ApiWorkflowClient, "export_label_box_data_rows_by_tag_id"
    )
    mocked_warning = mocker.patch("warnings.warn")
    client = ApiWorkflowClient()
    client._dataset_id = dataset_id
    client.export_label_box_data_rows_by_tag_name(tag_name)
    mocked_get_tag.assert_called_once_with(tag_name)
    mocked_export.assert_called_once_with(tag.id)
    warning_text = str(mocked_warning.call_args[0][0])
    assert warning_text == (
        "This method exports data in the deprecated Labelbox v3 format and "
        "will be removed in the future. Use export_label_box_v4_data_rows_by_tag_name "
        "to export data in the Labelbox v4 format instead."
    )


def test_export_label_box_v4_data_rows_by_tag_id(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_paginate = mocker.patch.object(api_workflow_export, "paginate_endpoint")
    mocked_api = mocker.MagicMock()

    client = ApiWorkflowClient()
    client._dataset_id = generate_id()
    client._tags_api = mocked_api
    client.export_label_box_v4_data_rows_by_tag_id(tag_id="tag_id")
    mocked_paginate.assert_called_once()
    call_args = mocked_paginate.call_args[0]
    assert call_args[0] == mocked_api.export_tag_to_label_box_v4_data_rows


def test_export_label_box_v4_data_rows_by_tag_name(mocker: MockerFixture) -> None:
    dataset_id = generate_id()
    tag_name = "some-tag"
    tag = _get_tag(dataset_id=dataset_id, tag_name=tag_name)
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_get_tag = mocker.patch.object(
        ApiWorkflowClient, "get_tag_by_name", return_value=tag
    )
    mocked_export = mocker.patch.object(
        ApiWorkflowClient, "export_label_box_v4_data_rows_by_tag_id"
    )
    client = ApiWorkflowClient()
    client._dataset_id = dataset_id
    client.export_label_box_v4_data_rows_by_tag_name(tag_name)
    mocked_get_tag.assert_called_once_with(tag_name)
    mocked_export.assert_called_once_with(tag.id)


def test_export_label_studio_tasks_by_tag_name(mocker: MockerFixture) -> None:
    dataset_id = generate_id()
    tag_name = "some-tag"
    tag = _get_tag(dataset_id=dataset_id, tag_name=tag_name)
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocked_get_tag = mocker.patch.object(
        ApiWorkflowClient, "get_tag_by_name", return_value=tag
    )
    mocked_export = mocker.patch.object(
        ApiWorkflowClient, "export_label_studio_tasks_by_tag_id"
    )
    client = ApiWorkflowClient()
    client._dataset_id = dataset_id
    client.export_label_studio_tasks_by_tag_name(tag_name)
    mocked_get_tag.assert_called_once_with(tag_name)
    mocked_export.assert_called_once_with(tag.id)
