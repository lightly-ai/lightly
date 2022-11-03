import json
import random
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from typing import Any, List
from unittest import mock

from lightly.api.api_workflow_compute_worker import (
    STATE_SCHEDULED_ID_NOT_FOUND,
    ArtifactNotExist,
    ComputeWorkerRunInfo,
)
from lightly.openapi_generated.swagger_client import (
    SelectionConfig,
    SelectionConfigEntry,
    SelectionInputType,
    SelectionStrategyType,
    ApiClient,
    DockerApi,
    SelectionConfigEntryInput,
    SelectionStrategyThresholdOperation,
    SelectionInputPredictionsName,
    SelectionConfigEntryStrategy,
    DockerWorkerConfig,
    DockerWorkerType,
    DockerRunArtifactData,
    DockerRunScheduledPriority,
    DockerRunScheduledState,
    DockerRunState,
)
from lightly.openapi_generated.swagger_client.models.docker_run_artifact_type import DockerRunArtifactType
from lightly.openapi_generated.swagger_client.models.docker_run_data import (
    DockerRunData,
)
from lightly.openapi_generated.swagger_client.models.docker_run_scheduled_data import (
    DockerRunScheduledData,
)
from lightly.openapi_generated.swagger_client.rest import ApiException
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup
from lightly.api import api_workflow_compute_worker, ApiWorkflowClient


class TestApiWorkflowComputeWorker(MockedApiWorkflowSetup):
    def test_register_compute_worker(self):
        # default name
        worker_id = self.api_workflow_client.register_compute_worker()
        assert worker_id
        # custom name
        worker_id = self.api_workflow_client.register_compute_worker(name="my-worker")
        assert worker_id

    def test_delete_compute_worker(self):
        worker_id = self.api_workflow_client.register_compute_worker(name="my-worker")
        assert worker_id
        self.api_workflow_client.delete_compute_worker(worker_id)

    def test_create_compute_worker_config(self):
        config_id = self.api_workflow_client.create_compute_worker_config(
            worker_config={
                "enable_corruptness_check": True,
                "stopping_condition": {
                    "n_samples": 10,
                },
            },
            lightly_config={
                "resize": 224,
                "loader": {
                    "batch_size": 64,
                },
            },
            selection_config={
                "n_samples": 20,
                "strategies": [
                    {
                        "input": {
                            "type": "EMBEDDINGS",
                            "dataset_id": "some-dataset-id",
                            "tag_name": "some-tag-name",
                        },
                        "strategy": {"type": "SIMILARITY"},
                    },
                ],
            },
        )
        assert config_id

    def test_schedule_compute_worker_run(self):
        scheduled_run_id = self.api_workflow_client.schedule_compute_worker_run(
            worker_config={
                "enable_corruptness_check": True,
                "stopping_condition": {
                    "n_samples": 10,
                },
            },
            lightly_config={
                "resize": 224,
                "loader": {
                    "batch_size": 64,
                },
            },
        )
        assert scheduled_run_id

    def test_schedule_compute_worker_run__priority(self):
        scheduled_run_id = self.api_workflow_client.schedule_compute_worker_run(
            worker_config={
            },
            lightly_config={
            },
            priority=DockerRunScheduledPriority.HIGH
        )
        assert scheduled_run_id

    def test_schedule_compute_worker_run__runs_on(self):
        scheduled_run_id = self.api_workflow_client.schedule_compute_worker_run(
            worker_config={
            },
            lightly_config={
            },
            runs_on=["AAA", "BBB"]
        )
        assert scheduled_run_id

    def test_get_compute_worker_ids(self):
        ids = self.api_workflow_client.get_compute_worker_ids()
        assert all(isinstance(id_, str) for id_ in ids)

    def test_get_compute_worker_runs(self):
        runs = self.api_workflow_client.get_compute_worker_runs()
        assert len(runs) > 0
        assert all(isinstance(run, DockerRunData) for run in runs)

    def test_get_scheduled_compute_worker_runs(self):
        runs = self.api_workflow_client.get_scheduled_compute_worker_runs()
        dataset_id = self.api_workflow_client.dataset_id
        assert len(runs) > 0
        assert all(isinstance(run, DockerRunScheduledData) for run in runs)
        assert all(run.dataset_id == dataset_id for run in runs)

    def _check_if_openapi_generated_obj_is_valid(self, obj) -> Any:
        api_client = ApiClient()

        obj_as_json = json.dumps(api_client.sanitize_for_serialization(obj))

        mocked_response = mock.MagicMock()
        mocked_response.data = obj_as_json
        obj_api = api_client.deserialize(mocked_response, type(obj).__name__)

        self.assertDictEqual(obj.to_dict(), obj_api.to_dict())

        return obj_api

    def test_selection_config(self):
        selection_config = SelectionConfig(
            n_samples=1,
            strategies=[
                SelectionConfigEntry(
                    input=SelectionConfigEntryInput(type=SelectionInputType.EMBEDDINGS),
                    strategy=SelectionConfigEntryStrategy(
                        type=SelectionStrategyType.DIVERSITY,
                        stopping_condition_minimum_distance=-1,
                    ),
                ),
                SelectionConfigEntry(
                    input=SelectionConfigEntryInput(
                        type=SelectionInputType.SCORES,
                        task="my-classification-task",
                        score="uncertainty_margin",
                    ),
                    strategy=SelectionConfigEntryStrategy(
                        type=SelectionStrategyType.WEIGHTS
                    ),
                ),
                SelectionConfigEntry(
                    input=SelectionConfigEntryInput(
                        type=SelectionInputType.METADATA, key="lightly.sharpness"
                    ),
                    strategy=SelectionConfigEntryStrategy(
                        type=SelectionStrategyType.THRESHOLD,
                        threshold=20,
                        operation=SelectionStrategyThresholdOperation.BIGGER_EQUAL,
                    ),
                ),
                SelectionConfigEntry(
                    input=SelectionConfigEntryInput(
                        type=SelectionInputType.PREDICTIONS,
                        task="my_object_detection_task",
                        name=SelectionInputPredictionsName.CLASS_DISTRIBUTION,
                    ),
                    strategy=SelectionConfigEntryStrategy(
                        type=SelectionStrategyType.BALANCE,
                        target={"Ambulance": 0.2, "Bus": 0.4},
                    ),
                ),
            ],
        )
        config = DockerWorkerConfig(
            worker_type=DockerWorkerType.FULL, selection=selection_config
        )

        config_api = self._check_if_openapi_generated_obj_is_valid(config)


def test_selection_config_from_dict() -> None:
    cfg = {
        "n_samples": 10,
        "proportion_samples": 0.1,
        "strategies": [
            {
                "input": {
                    "type": "EMBEDDINGS",
                    "dataset_id": "some-dataset-id",
                    "tag_name": "some-tag-name",
                },
                "strategy": {"type": "SIMILARITY"},
            },
            {
                "input": {
                    "type": "METADATA",
                    "key": "lightly.sharpness",
                },
                "strategy": {
                    "type": "THRESHOLD",
                    "threshold": 20,
                    "operation": "BIGGER",
                },
            },
        ],
    }
    selection_cfg = api_workflow_compute_worker.selection_config_from_dict(cfg)
    assert selection_cfg.n_samples == 10
    assert selection_cfg.proportion_samples == 0.1
    assert selection_cfg.strategies is not None
    assert len(selection_cfg.strategies) == 2
    assert selection_cfg.strategies[0].input.type == "EMBEDDINGS"
    assert selection_cfg.strategies[0].input.dataset_id == "some-dataset-id"
    assert selection_cfg.strategies[0].input.tag_name == "some-tag-name"
    assert selection_cfg.strategies[0].strategy.type == "SIMILARITY"
    assert selection_cfg.strategies[1].input.type == "METADATA"
    assert selection_cfg.strategies[1].input.key == "lightly.sharpness"
    assert selection_cfg.strategies[1].strategy.type == "THRESHOLD"
    assert selection_cfg.strategies[1].strategy.threshold == 20
    assert selection_cfg.strategies[1].strategy.operation == "BIGGER"
    # verify that original dict was not mutated
    assert isinstance(cfg["strategies"][0]["input"], dict)


def test_selection_config_from_dict__missing_strategies() -> None:
    cfg = {}
    selection_cfg = api_workflow_compute_worker.selection_config_from_dict(cfg)
    assert selection_cfg.strategies == []


def test_selection_config_from_dict__extra_key() -> None:
    cfg = {"strategies": [], "invalid-key": 0}
    with pytest.raises(
        TypeError, match="got an unexpected keyword argument 'invalid-key'"
    ):
        api_workflow_compute_worker.selection_config_from_dict(cfg)


def test_selection_config_from_dict__extra_stratey_key() -> None:
    cfg = {
        "strategies": [
            {
                "input": {"type": "EMBEDDINGS"},
                "strategy": {"type": "DIVERSITY"},
                "invalid-key": {"type": ""},
            },
        ],
    }
    with pytest.raises(
        TypeError, match="got an unexpected keyword argument 'invalid-key'"
    ):
        api_workflow_compute_worker.selection_config_from_dict(cfg)


def test_selection_config_from_dict__extra_input_key() -> None:
    cfg = {
        "strategies": [
            {
                "input": {"type": "EMBEDDINGS", "datasetId": ""},
                "strategy": {"type": "DIVERSITY"},
            },
        ],
    }
    with pytest.raises(
        TypeError, match="got an unexpected keyword argument 'datasetId'"
    ):
        api_workflow_compute_worker.selection_config_from_dict(cfg)


def test_selection_config_from_dict__extra_strategy_strategy_key() -> None:
    cfg = {
        "strategies": [
            {
                "input": {"type": "EMBEDDINGS"},
                "strategy": {
                    "type": "DIVERSITY",
                    "stoppingConditionMinimumDistance": 0,
                },
            },
        ],
    }
    with pytest.raises(
        TypeError,
        match="got an unexpected keyword argument 'stoppingConditionMinimumDistance'",
    ):
        api_workflow_compute_worker.selection_config_from_dict(cfg)


def test_selection_config_from_dict__typo() -> None:
    cfg = {"nSamples": 10}
    with pytest.raises(
        TypeError, match="got an unexpected keyword argument 'nSamples'"
    ):
        api_workflow_compute_worker.selection_config_from_dict(cfg)


def test_get_scheduled_run_by_id() -> None:

    scheduled_runs = [
        DockerRunScheduledData(
            id=f"id_{i}",
            dataset_id="dataset_id",
            config_id="config_id",
            priority=DockerRunScheduledPriority,
            state=DockerRunScheduledState.OPEN,
            created_at=0,
            last_modified_at=1,
            runs_on=[]
        )
        for i in range(3)
    ]
    mocked_compute_worker_api = MagicMock(
        get_docker_runs_scheduled_by_dataset_id=lambda dataset_id: scheduled_runs
    )
    mocked_api_client = MagicMock(
        dataset_id="asdf", _compute_worker_api=mocked_compute_worker_api
    )

    scheduled_run_id = "id_2"
    scheduled_run_data = ApiWorkflowClient._get_scheduled_run_by_id(
        self=mocked_api_client, scheduled_run_id=scheduled_run_id
    )
    assert scheduled_run_data.id == scheduled_run_id


def test_get_scheduled_run_by_id_not_found() -> None:

    scheduled_runs = [
        DockerRunScheduledData(
            id=f"id_{i}",
            dataset_id="dataset_id",
            config_id="config_id",
            priority=DockerRunScheduledPriority,
            state=DockerRunScheduledState.OPEN,
            created_at=0,
            last_modified_at=1,
            runs_on=[]
        )
        for i in range(3)
    ]
    mocked_compute_worker_api = MagicMock(
        get_docker_runs_scheduled_by_dataset_id=lambda dataset_id: scheduled_runs
    )
    mocked_api_client = MagicMock(
        dataset_id="asdf", _compute_worker_api=mocked_compute_worker_api
    )

    scheduled_run_id = "id_5"
    with pytest.raises(
        ApiException,
        match=f"No scheduled run found for run with scheduled_run_id='{scheduled_run_id}'.",
    ):
        scheduled_run_data = ApiWorkflowClient._get_scheduled_run_by_id(
            self=mocked_api_client, scheduled_run_id=scheduled_run_id
        )


def test_get_compute_worker_state_and_message_OPEN() -> None:

    scheduled_run = DockerRunScheduledData(
        id=f"id_2",
        dataset_id="dataset_id",
        config_id="config_id",
        priority=DockerRunScheduledPriority,
        state=DockerRunScheduledState.OPEN,
        created_at=0,
        last_modified_at=1,
        runs_on=[6]
    )

    def mocked_raise_exception(*args, **kwargs):
        raise ApiException

    mocked_api_client = MagicMock(
        dataset_id="asdf",
        _compute_worker_api=MagicMock(
            get_docker_run_by_scheduled_id=mocked_raise_exception
        ),
        _get_scheduled_run_by_id=lambda id: scheduled_run,
    )

    run_info = ApiWorkflowClient.get_compute_worker_run_info(
        self=mocked_api_client, scheduled_run_id=""
    )
    assert run_info.state == DockerRunScheduledState.OPEN
    assert run_info.message.startswith("Waiting for pickup by Lightly Worker.")
    assert run_info.in_end_state() == False


def test_get_compute_worker_state_and_message_CANCELED() -> None:
    def mocked_raise_exception(*args, **kwargs):
        raise ApiException

    mocked_api_client = MagicMock(
        dataset_id="asdf",
        _compute_worker_api=MagicMock(
            get_docker_run_by_scheduled_id=mocked_raise_exception
        ),
        _get_scheduled_run_by_id=mocked_raise_exception,
    )
    run_info = ApiWorkflowClient.get_compute_worker_run_info(
        self=mocked_api_client, scheduled_run_id=""
    )
    assert run_info.state == STATE_SCHEDULED_ID_NOT_FOUND
    assert run_info.message.startswith("Could not find a job for the given run_id:")
    assert run_info.in_end_state() == True


def test_get_compute_worker_state_and_message_docker_state() -> None:
    message = "SOME_MESSAGE"
    docker_run = DockerRunData(
        id="id",
        state=DockerRunState.GENERATING_REPORT,
        docker_version="",
        created_at=0,
        last_modified_at=0,
        message=message,
    )
    mocked_api_client = MagicMock(
        dataset_id="asdf",
        _compute_worker_api=MagicMock(
            get_docker_run_by_scheduled_id=lambda scheduled_id: docker_run
        ),
    )

    run_info = ApiWorkflowClient.get_compute_worker_run_info(
        self=mocked_api_client, scheduled_run_id=""
    )
    assert run_info.state == DockerRunState.GENERATING_REPORT
    assert run_info.message == message
    assert run_info.in_end_state() == False


def test_compute_worker_run_info_generator(mocker) -> None:

    states = [f"state_{i}" for i in range(7)]
    states[-1] = DockerRunState.COMPLETED

    class MockedApiWorkflowClient:
        def __init__(self, states: List[str]):
            self.states = states
            self.current_state_index = 0
            random.seed(42)

        def get_compute_worker_run_info(self, scheduled_run_id: str):
            state = self.states[self.current_state_index]
            if random.random() > 0.9:
                self.current_state_index += 1
            return ComputeWorkerRunInfo(state=state, message=state)

    mocker.patch("time.sleep", lambda _: None)

    mocked_client = MockedApiWorkflowClient(states)
    run_infos = list(
        ApiWorkflowClient.compute_worker_run_info_generator(
            mocked_client, scheduled_run_id=""
        )
    )

    expected_run_infos = [
        ComputeWorkerRunInfo(state=state, message=state) for state in states
    ]

    assert run_infos == expected_run_infos

def test_get_compute_worker_runs(mocker: MockerFixture) -> None:
    client = ApiWorkflowClient(token="123")
    mock_compute_worker_api = mocker.create_autospec(DockerApi, spec_set=True).return_value
    mock_compute_worker_api.get_docker_runs.return_value = [
        DockerRunData(id="run-1", created_at=20,  dataset_id="", docker_version="", state="", last_modified_at=0),
        DockerRunData(id="run-2", created_at=10,  dataset_id="", docker_version="", state="", last_modified_at=0),
    ]
    client._compute_worker_api = mock_compute_worker_api
    runs = client.get_compute_worker_runs()
    assert runs == [
        DockerRunData(id="run-2", created_at=10,  dataset_id="", docker_version="", state="", last_modified_at=0),
        DockerRunData(id="run-1", created_at=20,  dataset_id="", docker_version="", state="", last_modified_at=0),
    ]

def test_get_compute_worker_runs__dataset(mocker: MockerFixture) -> None:
    client = ApiWorkflowClient(token="123")
    mock_compute_worker_api = mocker.create_autospec(DockerApi, spec_set=True).return_value
    mock_compute_worker_api.get_docker_runs.return_value = [
        DockerRunData(id="run-1", dataset_id="dataset-1", docker_version="", state="", created_at=0, last_modified_at=0),
        DockerRunData(id="run-2", dataset_id="dataset-2", docker_version="", state="", created_at=0, last_modified_at=0),
    ]
    client._compute_worker_api = mock_compute_worker_api
    runs = client.get_compute_worker_runs(dataset_id="dataset-2")
    assert runs == [
        DockerRunData(id="run-2", dataset_id="dataset-2", docker_version="", state="", created_at=0, last_modified_at=0),
    ]

def test_download_compute_worker_run_artifacts(mocker: MockerFixture) -> None:
    client = ApiWorkflowClient(token="123")
    mock_download_compute_worker_run_artifact = mocker.MagicMock(
        spec_set=client._download_compute_worker_run_artifact
    )
    client._download_compute_worker_run_artifact = mock_download_compute_worker_run_artifact
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
        mocker.call(run_id="run-1", artifact_id="artifact-1", output_path="output_dir/report.pdf", timeout=60),
        mocker.call(run_id="run-1", artifact_id="artifact-2", output_path="output_dir/checkpoint.ckpt", timeout=60),
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
    client._download_compute_worker_run_artifact = mock_download_compute_worker_run_artifact
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
    client._download_compute_worker_run_artifact = mock_download_compute_worker_run_artifact
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
    client._download_compute_worker_run_artifact = mock_download_compute_worker_run_artifact
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
