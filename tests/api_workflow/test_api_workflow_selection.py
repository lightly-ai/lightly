from lightly.active_learning.config.selection_config import SelectionConfig, \
    SamplingConfig
from lightly.openapi_generated.swagger_client import TagData, SamplingMethod
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup


class TestApiWorkflowSelection(MockedApiWorkflowSetup):

    def test_sampling_deprecated(self):
        self.api_workflow_client.embedding_id = "embedding_id_xyz"

        with self.assertWarns(PendingDeprecationWarning):
            sampling_config = SamplingConfig(SamplingMethod.CORESET, n_samples=32)

        with self.assertWarns(PendingDeprecationWarning):
            new_tag_data = self.api_workflow_client.sampling(selection_config=sampling_config)
        assert isinstance(new_tag_data, TagData)

    def test_selection(self):
        self.api_workflow_client.embedding_id = "embedding_id_xyz"

        selection_config = SelectionConfig()

        new_tag_data = self.api_workflow_client.selection(selection_config=selection_config)
        assert isinstance(new_tag_data, TagData)

    def test_runtime_error_on_existing_tag_name(self):
        self.api_workflow_client.embedding_id = "embedding_id_xyz"

        selection_config = SelectionConfig(name='initial-tag')

        with self.assertRaises(RuntimeError):
            new_tag_data = self.api_workflow_client.selection(selection_config=selection_config)
