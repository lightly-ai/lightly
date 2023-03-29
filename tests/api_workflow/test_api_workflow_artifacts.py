import pytest
from pytest_mock import MockerFixture

from lightly.api import ApiWorkflowClient, ArtifactNotExist
from lightly.openapi_generated.swagger_client import (
    DockerApi,
    DockerRunArtifactData,
    DockerRunArtifactType,
    DockerRunData,
)


def test_download_compute_worker_run_artifacts(mocker: MockerFixture) -> None:
    client = ApiWorkflowClient(token="123")
    mock_download_compute_worker_run_artifact = mocker.MagicMock(
        spec_set=client._download_compute_worker_run_artifact
    )
    client._download_compute_worker_run_artifact = (
        mock_download_compute_worker_run_artifact
    )
    run = DockerRunData(
        id="run-1",
        user_id="user-id",
        dataset_id="dataset-1",
        docker_version="",
        state="",
        created_at=0,
        last_modified_at=0,
        artifacts=[
            DockerRunArtifactData(
                id="artifact-1",
                file_name="report.pdf",
                type=DockerRunArtifactType.REPORT_PDF,
            ),
            DockerRunArtifactData(
                id="artifact-2",
                file_name="checkpoint.ckpt",
                type=DockerRunArtifactType.CHECKPOINT,
            ),
        ],
    )
    client.download_compute_worker_run_artifacts(run=run, output_dir="output_dir")
    calls = [
        mocker.call(
            run_id="run-1",
            artifact_id="artifact-1",
            output_path="output_dir/report.pdf",
            timeout=60,
        ),
        mocker.call(
            run_id="run-1",
            artifact_id="artifact-2",
            output_path="output_dir/checkpoint.ckpt",
            timeout=60,
        ),
    ]
    mock_download_compute_worker_run_artifact.assert_has_calls(calls=calls)
    assert mock_download_compute_worker_run_artifact.call_count == len(calls)


def test__download_compute_worker_run_artifact_by_type(
    mocker: MockerFixture,
) -> None:
    client = ApiWorkflowClient(token="123")
    mock_download_compute_worker_run_artifact = mocker.MagicMock(
        spec_set=client._download_compute_worker_run_artifact
    )
    client._download_compute_worker_run_artifact = (
        mock_download_compute_worker_run_artifact
    )
    run = DockerRunData(
        id="run-1",
        user_id="user-id",
        dataset_id="dataset-1",
        docker_version="",
        state="",
        created_at=0,
        last_modified_at=0,
        artifacts=[
            DockerRunArtifactData(
                id="artifact-1",
                file_name="report.pdf",
                type=DockerRunArtifactType.REPORT_PDF,
            ),
            DockerRunArtifactData(
                id="artifact-2",
                file_name="checkpoint.ckpt",
                type=DockerRunArtifactType.CHECKPOINT,
            ),
        ],
    )
    client._download_compute_worker_run_artifact_by_type(
        run=run,
        artifact_type=DockerRunArtifactType.CHECKPOINT,
        output_path="output_dir/checkpoint.ckpt",
        timeout=0,
    )
    mock_download_compute_worker_run_artifact.assert_called_once_with(
        run_id="run-1",
        artifact_id="artifact-2",
        output_path="output_dir/checkpoint.ckpt",
        timeout=0,
    )


def test__download_compute_worker_run_artifact_by_type__no_artifacts(
    mocker: MockerFixture,
) -> None:
    client = ApiWorkflowClient(token="123")
    mock_download_compute_worker_run_artifact = mocker.MagicMock(
        spec_set=client._download_compute_worker_run_artifact
    )
    client._download_compute_worker_run_artifact = (
        mock_download_compute_worker_run_artifact
    )
    run = DockerRunData(
        id="run-1",
        user_id="user-id",
        dataset_id="dataset-1",
        docker_version="",
        state="",
        created_at=0,
        last_modified_at=0,
        artifacts=None,
    )
    with pytest.raises(ArtifactNotExist, match="Run has no artifacts."):
        client._download_compute_worker_run_artifact_by_type(
            run=run,
            artifact_type=DockerRunArtifactType.CHECKPOINT,
            output_path="output_dir/checkpoint.ckpt",
            timeout=0,
        )


def test__download_compute_worker_run_artifact_by_type__no_artifact_with_type(
    mocker: MockerFixture,
) -> None:
    client = ApiWorkflowClient(token="123")
    mock_download_compute_worker_run_artifact = mocker.MagicMock(
        spec_set=client._download_compute_worker_run_artifact
    )
    client._download_compute_worker_run_artifact = (
        mock_download_compute_worker_run_artifact
    )
    run = DockerRunData(
        id="run-1",
        user_id="user-id",
        dataset_id="dataset-1",
        docker_version="",
        state="",
        created_at=0,
        last_modified_at=0,
        artifacts=[
            DockerRunArtifactData(
                id="artifact-1",
                file_name="report.pdf",
                type=DockerRunArtifactType.REPORT_PDF,
            ),
        ],
    )
    with pytest.raises(ArtifactNotExist, match="No artifact with type"):
        client._download_compute_worker_run_artifact_by_type(
            run=run,
            artifact_type=DockerRunArtifactType.CHECKPOINT,
            output_path="output_dir/checkpoint.ckpt",
            timeout=0,
        )


def test__get_compute_worker_run_checkpoint_url(
    mocker: MockerFixture,
) -> None:
    mocked_client = mocker.MagicMock(spec=ApiWorkflowClient)
    mocked_artifact = mocker.MagicMock(spec_set=DockerRunArtifactData)
    mocked_client._get_artifact_by_type.return_value = mocked_artifact
    mocked_client._compute_worker_api = mocker.MagicMock(spec_set=DockerApi)
    mocked_client._compute_worker_api.get_docker_run_artifact_read_url_by_id.return_value = (
        "some_read_url"
    )

    mocked_run = mocker.MagicMock(spec_set=DockerRunData)
    read_url = ApiWorkflowClient.get_compute_worker_run_checkpoint_url(
        self=mocked_client, run=mocked_run
    )

    assert read_url == "some_read_url"
    mocked_client._get_artifact_by_type.assert_called_with(
        artifact_type=DockerRunArtifactType.CHECKPOINT, run=mocked_run
    )
    mocked_client._compute_worker_api.get_docker_run_artifact_read_url_by_id.assert_called_with(
        run_id=mocked_run.id, artifact_id=mocked_artifact.id
    )
