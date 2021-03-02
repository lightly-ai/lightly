from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.openapi_generated.swagger_client import TagData
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup


class TestApiWorkflowSampling(MockedApiWorkflowSetup):
    def test_sampling(self):
        self.api_workflow_client.embedding_id = "embedding_id_xyz"

        sampler_config = SamplerConfig()

        new_tag_data = self.api_workflow_client.sampling(sampler_config=sampler_config)
        assert isinstance(new_tag_data, TagData)

