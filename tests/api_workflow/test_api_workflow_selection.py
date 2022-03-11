from lightly.active_learning.config.selection_config import SelectionConfig
from lightly.openapi_generated.swagger_client import TagData
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup


class TestApiWorkflowSelection(MockedApiWorkflowSetup):
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
