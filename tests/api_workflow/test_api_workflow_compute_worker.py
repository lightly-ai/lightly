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