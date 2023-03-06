import pytest
from pytest_mock import MockerFixture

from lightly.openapi_generated.swagger_client import (
    DockerRunArtifactData,
    DockerRunArtifactType,
    DockerRunData,
)
from lightly.api import ApiWorkflowClient, ArtifactNotExist


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
