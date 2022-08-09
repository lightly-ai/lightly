import json
import unittest
from typing import Dict
from unittest import mock

from lightly.openapi_generated.swagger_client import DockerWorkerSelectionConfig, DockerWorkerSelectionConfigEntry, DockerWorkerSelectionInputType, \
    DockerWorkerSelectionStrategyType, ApiClient, DockerApi
from lightly.openapi_generated.swagger_client.models.docker_run_data import DockerRunData
from lightly.openapi_generated.swagger_client.models.docker_run_scheduled_data import DockerRunScheduledData
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup


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

    def _check_if_openapi_generated_obj_is_valid(self, obj):
        api_client = ApiClient()

        obj_as_json = json.dumps(api_client.sanitize_for_serialization(obj))

        mocked_response = mock.MagicMock()
        mocked_response.data = obj_as_json
        obj_api = api_client.deserialize(mocked_response, type(obj).__name__)

        self.assertDictEqual(obj.to_dict(), obj_api.to_dict())

    @unittest.skip("This fails due to problematic API specs")
    def test_selection_config(self):
        selection_config = DockerWorkerSelectionConfig(
            n_samples=1,
            strategies=[
                DockerWorkerSelectionConfigEntry(
                    input={"type": DockerWorkerSelectionInputType.EMBEDDINGS},
                    strategy={"type": DockerWorkerSelectionStrategyType.DIVERSIFY, "stopping_condition_minimum_distance": -1}
                ),
                DockerWorkerSelectionConfigEntry(
                    input={"type": DockerWorkerSelectionInputType.SCORES, "task": "my-classification-task", "score": "uncertainty_margin"},
                    strategy={"type": DockerWorkerSelectionStrategyType.WEIGHTS}
                ),
                DockerWorkerSelectionConfigEntry(
                    input={"type": DockerWorkerSelectionInputType.METADATA, "key": "lightly.sharpness"},
                    strategy={"type": DockerWorkerSelectionStrategyType.THRESHOLD, "threshold": 20, "operation": "BIGGER_EQUAL"}
                ),
                DockerWorkerSelectionConfigEntry(
                    input={"type": DockerWorkerSelectionInputType.PREDICTIONS, "task": "my_object_detection_task", "name": "CLASS_DISTRIBUTION"},
                    strategy={"type": DockerWorkerSelectionStrategyType.BALANCE, "target": {"Ambulance": 0.2, "Bus": 0.4}}
                )
            ]
        )

        self._check_if_openapi_generated_obj_is_valid(selection_config)
