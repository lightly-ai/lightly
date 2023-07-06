import pytest
import pytest_mock

from lightly import api as lightly_api
from lightly.openapi_generated.swagger_client import api as swagger_api
from lightly.openapi_generated.swagger_client import models
from tests.api_workflow import utils


def test_download_compute_worker_run_artifacts(
    mocker: pytest_mock.MockerFixture,
) -> None:
    client = lightly_api.ApiWorkflowClient(token="123")
    mock_download_compute_worker_run_artifact = mocker.MagicMock(
        spec_set=client._download_compute_worker_run_artifact
    )
    client._download_compute_worker_run_artifact = (
        mock_download_compute_worker_run_artifact
    )
    run_id = utils.generate_id()
    artifact_ids = [utils.generate_id(), utils.generate_id()]
    run = models.DockerRunData(
        id=run_id,
        user_id="user-id",
        dataset_id=utils.generate_id(),
        docker_version="",
        state=models.DockerRunState.COMPUTING_METADATA,
        created_at=0,
        last_modified_at=0,
        artifacts=[
            models.DockerRunArtifactData(
                id=artifact_ids[0],
                file_name="report.pdf",
                type=models.DockerRunArtifactType.REPORT_PDF,
            ),
            models.DockerRunArtifactData(
                id=artifact_ids[1],
                file_name="checkpoint.ckpt",
                type=models.DockerRunArtifactType.CHECKPOINT,
            ),
        ],
    )
    client.download_compute_worker_run_artifacts(run=run, output_dir="output_dir")
    calls = [
        mocker.call(
            run_id=run_id,
            artifact_id=artifact_ids[0],
            output_path="output_dir/report.pdf",
            timeout=60,
        ),
        mocker.call(
            run_id=run_id,
            artifact_id=artifact_ids[1],
            output_path="output_dir/checkpoint.ckpt",
            timeout=60,
        ),
    ]
    mock_download_compute_worker_run_artifact.assert_has_calls(calls=calls)
    assert mock_download_compute_worker_run_artifact.call_count == len(calls)


def test__download_compute_worker_run_artifact_by_type(
    mocker: pytest_mock.MockerFixture,
) -> None:
    client = lightly_api.ApiWorkflowClient(token="123")
    mock_download_compute_worker_run_artifact = mocker.MagicMock(
        spec_set=client._download_compute_worker_run_artifact
    )
    client._download_compute_worker_run_artifact = (
        mock_download_compute_worker_run_artifact
    )
    run_id = utils.generate_id()
    artifact_ids = [utils.generate_id(), utils.generate_id()]
    run = models.DockerRunData(
        id=run_id,
        user_id="user-id",
        dataset_id=utils.generate_id(),
        docker_version="",
        state=models.DockerRunState.COMPUTING_METADATA,
        created_at=0,
        last_modified_at=0,
        artifacts=[
            models.DockerRunArtifactData(
                id=artifact_ids[0],
                file_name="report.pdf",
                type=models.DockerRunArtifactType.REPORT_PDF,
            ),
            models.DockerRunArtifactData(
                id=artifact_ids[1],
                file_name="checkpoint.ckpt",
                type=models.DockerRunArtifactType.CHECKPOINT,
            ),
        ],
    )
    client._download_compute_worker_run_artifact_by_type(
        run=run,
        artifact_type=models.DockerRunArtifactType.CHECKPOINT,
        output_path="output_dir/checkpoint.ckpt",
        timeout=0,
    )
    mock_download_compute_worker_run_artifact.assert_called_once_with(
        run_id=run_id,
        artifact_id=artifact_ids[1],
        output_path="output_dir/checkpoint.ckpt",
        timeout=0,
    )


def test__download_compute_worker_run_artifact_by_type__no_artifacts(
    mocker: pytest_mock.MockerFixture,
) -> None:
    client = lightly_api.ApiWorkflowClient(token="123")
    mock_download_compute_worker_run_artifact = mocker.MagicMock(
        spec_set=client._download_compute_worker_run_artifact
    )
    client._download_compute_worker_run_artifact = (
        mock_download_compute_worker_run_artifact
    )
    run = models.DockerRunData(
        id=utils.generate_id(),
        user_id="user-id",
        dataset_id=utils.generate_id(),
        docker_version="",
        state=models.DockerRunState.COMPUTING_METADATA,
        created_at=0,
        last_modified_at=0,
        artifacts=None,
    )
    with pytest.raises(lightly_api.ArtifactNotExist, match="Run has no artifacts."):
        client._download_compute_worker_run_artifact_by_type(
            run=run,
            artifact_type=models.DockerRunArtifactType.CHECKPOINT,
            output_path="output_dir/checkpoint.ckpt",
            timeout=0,
        )


def test__download_compute_worker_run_artifact_by_type__no_artifact_with_type(
    mocker: pytest_mock.MockerFixture,
) -> None:
    client = lightly_api.ApiWorkflowClient(token="123")
    mock_download_compute_worker_run_artifact = mocker.MagicMock(
        spec_set=client._download_compute_worker_run_artifact
    )
    client._download_compute_worker_run_artifact = (
        mock_download_compute_worker_run_artifact
    )
    run = models.DockerRunData(
        id=utils.generate_id(),
        user_id="user-id",
        dataset_id=utils.generate_id(),
        docker_version="",
        state=models.DockerRunState.COMPUTING_METADATA,
        created_at=0,
        last_modified_at=0,
        artifacts=[
            models.DockerRunArtifactData(
                id=utils.generate_id(),
                file_name="report.pdf",
                type=models.DockerRunArtifactType.REPORT_PDF,
            ),
        ],
    )
    with pytest.raises(lightly_api.ArtifactNotExist, match="No artifact with type"):
        client._download_compute_worker_run_artifact_by_type(
            run=run,
            artifact_type=models.DockerRunArtifactType.CHECKPOINT,
            output_path="output_dir/checkpoint.ckpt",
            timeout=0,
        )


def test__get_compute_worker_run_checkpoint_url(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocked_client = mocker.MagicMock(spec=lightly_api.ApiWorkflowClient)
    mocked_artifact = models.DockerRunArtifactData(
        id=utils.generate_id(),
        file_name="report.pdf",
        type=models.DockerRunArtifactType.REPORT_PDF,
    )
    mocked_client._get_artifact_by_type.return_value = mocked_artifact
    mocked_client._compute_worker_api = mocker.MagicMock(spec_set=swagger_api.DockerApi)
    mocked_client._compute_worker_api.get_docker_run_artifact_read_url_by_id.return_value = (
        "some_read_url"
    )

    run = models.DockerRunData(
        id=utils.generate_id(),
        user_id="user-id",
        dataset_id=utils.generate_id(),
        docker_version="",
        state=models.DockerRunState.COMPUTING_METADATA,
        created_at=0,
        last_modified_at=0,
        artifacts=[mocked_artifact],
    )
    read_url = lightly_api.ApiWorkflowClient.get_compute_worker_run_checkpoint_url(
        self=mocked_client, run=run
    )

    assert read_url == "some_read_url"
    mocked_client._get_artifact_by_type.assert_called_with(
        artifact_type=models.DockerRunArtifactType.CHECKPOINT, run=run
    )
    mocked_client._compute_worker_api.get_docker_run_artifact_read_url_by_id.assert_called_with(
        run_id=run.id, artifact_id=mocked_artifact.id
    )
