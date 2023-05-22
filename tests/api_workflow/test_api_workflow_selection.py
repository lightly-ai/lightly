import pytest
from pytest_mock import MockerFixture

from lightly.active_learning.config.selection_config import (
    SamplingConfig,
    SelectionConfig,
)
from lightly.api import ApiWorkflowClient, api_workflow_selection
from lightly.openapi_client.models import (
    JobResultType,
    JobState,
    JobStatusData,
    JobStatusDataResult,
    SamplingCreateRequest,
    SamplingMethod,
    TagData,
)
from tests.api_workflow.utils import generate_id


def _get_tags(dataset_id: str, tag_name: str = "just-a-tag") -> list[TagData]:
    return [
        TagData(
            id=generate_id(),
            dataset_id=dataset_id,
            prev_tag_id=None,
            bit_mask_data="0x1",
            name=tag_name,
            tot_size=4,
            created_at=1577836800,
            changes=[],
        )
    ]


def _get_sampling_create_request(tag_name: str = "new-tag") -> SamplingCreateRequest:
    return SamplingCreateRequest(
        new_tag_name=tag_name,
        method=SamplingMethod.RANDOM,
        config={},
    )


def test_sampling_deprecated(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(ApiWorkflowClient, "selection")
    mocked_warning = mocker.patch("warnings.warn")
    client = ApiWorkflowClient()
    client.sampling()
    mocked_warning.assert_called_once()


def test_selection__tag_exists(mocker: MockerFixture) -> None:
    tag_name = "some-tag"
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(
        ApiWorkflowClient,
        "get_all_tags",
        return_value=_get_tags(dataset_id=generate_id(), tag_name=tag_name),
    )

    client = ApiWorkflowClient()
    with pytest.raises(RuntimeError) as exception:
        client.selection(selection_config=SelectionConfig(name=tag_name))

        assert (
            str(exception.value) == "There already exists a tag with tag_name some-tag"
        )


def test_selection__no_tags(mocker: MockerFixture) -> None:
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(ApiWorkflowClient, "get_all_tags", return_value=[])

    client = ApiWorkflowClient()
    with pytest.raises(RuntimeError) as exception:
        client.selection(selection_config=SelectionConfig(name="some-tag"))

        assert str(exception.value) == "There exists no initial-tag for this dataset."


def test_selection(mocker: MockerFixture) -> None:
    tag_name = "some-tag"
    dataset_id = generate_id()
    mocker.patch("time.sleep")
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(
        ApiWorkflowClient, "get_all_tags", return_value=_get_tags(dataset_id=dataset_id)
    )

    mocker.patch.object(
        ApiWorkflowClient,
        "_create_selection_create_request",
        return_value=_get_sampling_create_request(),
    )

    mocked_selection_api = mocker.MagicMock()
    mocked_sampling_response = mocker.MagicMock()
    mocked_sampling_response.job_id = generate_id()
    mocked_selection_api.trigger_sampling_by_id.return_value = mocked_sampling_response

    mocked_jobs_api = mocker.MagicMock()
    mocked_get_job_status = mocker.MagicMock(
        return_value=JobStatusData(
            id=generate_id(),
            wait_time_till_next_poll=1,
            created_at=0,
            status=JobState.FINISHED,
            result=JobStatusDataResult(type=JobResultType.SAMPLING, data="new-tag-id"),
        )
    )
    mocked_jobs_api.get_job_status_by_id = mocked_get_job_status

    mocked_tags_api = mocker.MagicMock()

    client = ApiWorkflowClient()
    client._selection_api = mocked_selection_api
    client._jobs_api = mocked_jobs_api
    client._tags_api = mocked_tags_api
    client._dataset_id = dataset_id
    client.embedding_id = "embedding-id"
    client.selection(selection_config=SelectionConfig(name=tag_name))

    mocked_get_job_status.assert_called_once()
    mocked_tags_api.get_tag_by_tag_id.assert_called_once_with(
        dataset_id=dataset_id, tag_id="new-tag-id"
    )


def test_selection__job_failed(mocker: MockerFixture) -> None:
    dataset_id = generate_id()
    job_id = "some-job-id"
    mocker.patch("time.sleep")
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(
        ApiWorkflowClient, "get_all_tags", return_value=_get_tags(dataset_id=dataset_id)
    )

    mocker.patch.object(
        ApiWorkflowClient,
        "_create_selection_create_request",
        return_value=_get_sampling_create_request(),
    )

    mocked_selection_api = mocker.MagicMock()
    mocked_sampling_response = mocker.MagicMock()
    mocked_sampling_response.job_id = job_id
    mocked_selection_api.trigger_sampling_by_id.return_value = mocked_sampling_response

    mocked_jobs_api = mocker.MagicMock()
    mocked_get_job_status = mocker.MagicMock(
        return_value=JobStatusData(
            id=generate_id(),
            wait_time_till_next_poll=1,
            created_at=0,
            status=JobState.FAILED,
            error="bad job",
        )
    )
    mocked_jobs_api.get_job_status_by_id = mocked_get_job_status

    client = ApiWorkflowClient()
    client._selection_api = mocked_selection_api
    client._jobs_api = mocked_jobs_api
    client._dataset_id = dataset_id
    client.embedding_id = "embedding-id"
    with pytest.raises(RuntimeError) as exception:
        client.selection(selection_config=SelectionConfig(name="some-tag"))
        assert str(exception.value) == (
            "Selection job with job_id some-job-id failed with error bad job"
        )


def test_selection__too_many_errors(mocker: MockerFixture) -> None:
    dataset_id = generate_id()
    job_id = "some-job-id"
    mocker.patch("time.sleep")
    mocked_print = mocker.patch("builtins.print")
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(
        ApiWorkflowClient, "get_all_tags", return_value=_get_tags(dataset_id=dataset_id)
    )

    mocker.patch.object(
        ApiWorkflowClient,
        "_create_selection_create_request",
        return_value=_get_sampling_create_request(),
    )

    mocked_selection_api = mocker.MagicMock()
    mocked_sampling_response = mocker.MagicMock()
    mocked_sampling_response.job_id = job_id
    mocked_selection_api.trigger_sampling_by_id.return_value = mocked_sampling_response

    mocked_jobs_api = mocker.MagicMock()
    mocked_get_job_status = mocker.MagicMock(
        side_effect=[Exception("surprise!") for _ in range(20)]
    )
    mocked_jobs_api.get_job_status_by_id = mocked_get_job_status

    client = ApiWorkflowClient()
    client._selection_api = mocked_selection_api
    client._jobs_api = mocked_jobs_api
    client._dataset_id = dataset_id
    client.embedding_id = "embedding-id"
    with pytest.raises(Exception) as exception:
        client.selection(selection_config=SelectionConfig(name="some-tag"))
        assert str(exception.value) == "surprise!"
        mocked_print.assert_called_once_with(
            "Selection job with job_id some-job-id could not be started "
            "because of error: surprise!"
        )


def test_upload_scores(mocker: MockerFixture) -> None:
    dataset_id = generate_id()
    tags = _get_tags(dataset_id=dataset_id, tag_name="initial-tag")
    tag_id = tags[0].id
    mocker.patch.object(ApiWorkflowClient, "__init__", return_value=None)
    mocker.patch.object(
        ApiWorkflowClient,
        "get_all_tags",
        return_value=tags,
    )
    mocker.patch.object(
        api_workflow_selection, "_parse_active_learning_scores", return_value=[1]
    )
    mocked_api = mocker.MagicMock()
    mocked_create_score = mocked_api.create_or_update_active_learning_score_by_tag_id

    client = ApiWorkflowClient()
    client._scores_api = mocked_api
    client._dataset_id = dataset_id

    # without query_tag_id
    client.upload_scores(al_scores={"score_type": [1, 2, 3]})
    mocked_create_score.assert_called_once()
    kwargs = mocked_create_score.call_args[1]
    assert kwargs.get("tag_id") == tag_id

    # with query_tag_id
    mocked_create_score.reset_mock()
    client.upload_scores(
        al_scores={"score_type": [1, 2, 3]}, query_tag_id="some-tag-id"
    )
    mocked_create_score.assert_called_once()
    kwargs = mocked_create_score.call_args[1]
    assert kwargs.get("tag_id") == "some-tag-id"
