from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.openapi_generated.swagger_client import TagData
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup


class TestApiWorkflowDatasets(MockedApiWorkflowSetup):

    def test_create_dataset_new(self):
        self.api_workflow_client.datasets_api.reset()
        self.api_workflow_client.create_dataset(dataset_name="dataset_new")
        test_var = self.api_workflow_client.dataset_id

    def test_create_dataset_existing(self):
        self.api_workflow_client.datasets_api.reset()
        self.api_workflow_client.create_dataset(dataset_name="dataset_1")

    def test_create_dataset_with_counter(self):
        self.api_workflow_client.datasets_api.reset()
        self.api_workflow_client.create_dataset(dataset_name="basename")
        n_tries = 3
        for i in range(n_tries):
            self.api_workflow_client.create_new_dataset_with_unique_name(dataset_basename="basename")
        assert self.api_workflow_client.datasets_api.datasets[-1].name == f"basename_{n_tries}"

    def test_create_dataset_with_counter_nonexisting(self):
        self.api_workflow_client.datasets_api.reset()
        self.api_workflow_client.create_dataset(dataset_name="basename")
        self.api_workflow_client.create_new_dataset_with_unique_name(dataset_basename="baseName")
        assert self.api_workflow_client.datasets_api.datasets[-1].name == "baseName"

    def test_set_dataset_id_success(self):
        self.api_workflow_client.datasets_api.reset()
        self.api_workflow_client.set_dataset_id_by_name("dataset_1")
        assert self.api_workflow_client.dataset_id == "dataset_1_id"

    def test_set_dataset_id_error(self):
        self.api_workflow_client.datasets_api.reset()
        with self.assertRaises(ValueError):
            self.api_workflow_client.set_dataset_id_by_name("nonexisting-dataset")

    def test_delete_dataset(self):
        self.api_workflow_client.datasets_api.reset()
        self.api_workflow_client.create_dataset(dataset_name="dataset_to_delete")
        self.api_workflow_client.delete_dataset_by_id(self.api_workflow_client.dataset_id)
        assert not hasattr(self, "_dataset_id")


