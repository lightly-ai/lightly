import numpy as np

from lightly.active_learning.agents.agent import ActiveLearningAgent
from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.active_learning.scorers.classification import ScorerClassification
from lightly.openapi_generated.swagger_client import SamplingMethod
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup


class TestApiWorkflowAgent(MockedApiWorkflowSetup):
    def test_agent(self):
        self.api_workflow_client.embedding_id = "embedding_id_xyz"

        agent_0 = ActiveLearningAgent(self.api_workflow_client)
        agent_1 = ActiveLearningAgent(self.api_workflow_client, query_tag_name="query_tag_name_xyz")
        agent_2 = ActiveLearningAgent(self.api_workflow_client, query_tag_name="query_tag_name_xyz",
                                      preselected_tag_name="preselected_tag_name_xyz")
        agent_3 = ActiveLearningAgent(self.api_workflow_client, preselected_tag_name="preselected_tag_name_xyz")

        for agent in [agent_0, agent_1, agent_2, agent_3]:
            for batch_size in [2, 4]:
                n_samples = len(agent.labeled_set) + batch_size
                sampler_config = SamplerConfig(n_samples=n_samples)
                chosen_filenames = agent.query(sampler_config=sampler_config)

    def test_agent_with_scores(self):
        self.api_workflow_client.embedding_id = "embedding_id_xyz"

        agent = ActiveLearningAgent(self.api_workflow_client, preselected_tag_name="preselected_tag_name_xyz")

        for n_samples in [2, 6]:
            predictions = np.random.rand(len(agent.unlabeled_set), 10)
            predictions_normalized = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
            al_scorer = ScorerClassification(predictions_normalized)
            sampler_config = SamplerConfig(n_samples=n_samples, method=SamplingMethod.CORAL)
            chosen_filenames = agent.query(sampler_config=sampler_config, al_scorer=al_scorer)
