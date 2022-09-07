import json
import pytest
from typing import Dict, Any
from unittest import mock

from omegaconf import DictConfig, OmegaConf

from lightly.openapi_generated.swagger_client import SelectionConfig, SelectionConfigEntry, SelectionInputType, \
    SelectionStrategyType, ApiClient, DockerApi, SelectionConfigEntryInput, SelectionStrategyThresholdOperation, \
    SelectionInputPredictionsName, SelectionConfigEntryStrategy, DockerWorkerConfig, DockerWorkerType
from lightly.openapi_generated.swagger_client.models.docker_run_data import DockerRunData
from lightly.openapi_generated.swagger_client.models.docker_run_scheduled_data import DockerRunScheduledData
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup
from lightly.api import api_workflow_compute_worker

class TestApiWorkflowComputeWorker(MockedApiWorkflowSetup):
    def test_register_compute_worker(self):
        # default name
        worker_id = self.api_workflow_client.register_compute_worker()
        assert worker_id
        # custom name
        worker_id = self.api_workflow_client.register_compute_worker(name="my-worker")
        assert worker_id

    def test_delete_compute_worker(self):
        worker_id = self.api_workflow_client.register_compute_worker(name='my-worker')
        assert worker_id
        self.api_workflow_client.delete_compute_worker(worker_id)

    def test_create_compute_worker_config(self):
        config_id = self.api_workflow_client.create_compute_worker_config(
            worker_config={
                'enable_corruptness_check': True,
                'stopping_condition': {
                    'n_samples': 10,
                }
            },
            lightly_config={
                'resize': 224,
                'loader': {
                    'batch_size': 64,
                }
            },
            selection_config={
                'n_samples': 20,
                'strategies': [
                    {
                        "input": {"type": "EMBEDDINGS", "dataset_id": "some-dataset-id", "tag_name": "some-tag-name"},
                        "strategy": {"type": "SIMILARITY"}
                    },
                ]
            }
        )
        assert config_id

    def test_schedule_compute_worker_run(self):
        scheduled_run_id = self.api_workflow_client.schedule_compute_worker_run(
            worker_config={
                'enable_corruptness_check': True,
                'stopping_condition': {
                    'n_samples': 10,
                }
            },
            lightly_config={
                'resize': 224,
                'loader': {
                    'batch_size': 64,
                }
            }
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
                    strategy=SelectionConfigEntryStrategy(type=SelectionStrategyType.DIVERSITY, stopping_condition_minimum_distance=-1)
                ),
                SelectionConfigEntry(
                    input=SelectionConfigEntryInput(type=SelectionInputType.SCORES, task="my-classification-task", score="uncertainty_margin"),
                    strategy=SelectionConfigEntryStrategy(type=SelectionStrategyType.WEIGHTS)
                ),
                SelectionConfigEntry(
                    input=SelectionConfigEntryInput(type=SelectionInputType.METADATA, key="lightly.sharpness"),
                    strategy=SelectionConfigEntryStrategy(type=SelectionStrategyType.THRESHOLD, threshold=20, operation=SelectionStrategyThresholdOperation.BIGGER_EQUAL)
                ),
                SelectionConfigEntry(
                    input=SelectionConfigEntryInput(type=SelectionInputType.PREDICTIONS, task="my_object_detection_task", name=SelectionInputPredictionsName.CLASS_DISTRIBUTION),
                    strategy=SelectionConfigEntryStrategy(type=SelectionStrategyType.BALANCE, target= {"Ambulance": 0.2, "Bus": 0.4})
                )
            ]
        )
        config = DockerWorkerConfig(worker_type=DockerWorkerType.FULL, selection=selection_config)

        config_api = self._check_if_openapi_generated_obj_is_valid(config)

def test_selection_config_from_dict() -> None:
    cfg = {
        "n_samples": 10,
        "proportion_samples": 0.1,
        "strategies": [
            {
                "input": {"type": "EMBEDDINGS", "dataset_id": "some-dataset-id", "tag_name": "some-tag-name"},
                "strategy": {"type": "SIMILARITY"}
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
        ]
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
    assert isinstance(cfg['strategies'][0]['input'], dict)


def test_selection_config_from_dict__missing_strategies() -> None:
    cfg = {}
    selection_cfg = api_workflow_compute_worker.selection_config_from_dict(cfg)
    assert selection_cfg.strategies == []


def test_selection_config_from_dict__extra_key() -> None:
    cfg = {"strategies": [], "invalid-key": 0}
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'invalid-key'"):
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
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'invalid-key'"):
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
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'datasetId'"):
        api_workflow_compute_worker.selection_config_from_dict(cfg)


def test_selection_config_from_dict__extra_strategy_strategy_key() -> None:
    cfg = {
        "strategies": [
            {
                "input": {"type": "EMBEDDINGS"},
                "strategy": {"type": "DIVERSITY", "stoppingConditionMinimumDistance": 0},
            },
        ],
    }
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'stoppingConditionMinimumDistance'"):
        api_workflow_compute_worker.selection_config_from_dict(cfg)


def test_selection_config_from_dict__typo() -> None:
    cfg = {"nSamples": 10}
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'nSamples'"):
        api_workflow_compute_worker.selection_config_from_dict(cfg)
