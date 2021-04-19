import numpy as np

from lightly.active_learning.agents.agent import ActiveLearningAgent
from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.active_learning.scorers.classification import ScorerClassification
from lightly.openapi_generated.swagger_client import SamplingMethod
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowSetup


class TestActiveLearningAgent(MockedApiWorkflowSetup):
    def test_agent(self):
        self.api_workflow_client.embedding_id = "embedding_id_xyz"

        agent_0 = ActiveLearningAgent(self.api_workflow_client)
        agent_1 = ActiveLearningAgent(self.api_workflow_client, query_tag_name="query_tag_name_xyz")
        agent_2 = ActiveLearningAgent(self.api_workflow_client, query_tag_name="query_tag_name_xyz",
                                      preselected_tag_name="preselected_tag_name_xyz")
        agent_3 = ActiveLearningAgent(self.api_workflow_client, preselected_tag_name="preselected_tag_name_xyz")

        for method in [SamplingMethod.CORAL, SamplingMethod.CORESET, SamplingMethod.RANDOM]:
            for agent in [agent_0, agent_1, agent_2, agent_3]:
                for batch_size in [2, 6]:
                    n_old_labeled = len(agent.labeled_set)
                    n_old_unlabeled = len(agent.unlabeled_set)

                    n_samples = len(agent.labeled_set) + batch_size
                    if method == SamplingMethod.CORAL and len(agent.labeled_set) == 0:
                        sampler_config = SamplerConfig(n_samples=n_samples, method=SamplingMethod.CORESET)
                    else:
                        sampler_config = SamplerConfig(n_samples=n_samples, method=method)

                    if sampler_config.method == SamplingMethod.CORAL:
                        predictions = np.random.rand(len(agent.unlabeled_set), 10).astype(np.float32)
                        predictions_normalized = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
                        al_scorer = ScorerClassification(predictions_normalized)
                        labeled_set, added_set = agent.query(sampler_config=sampler_config, al_scorer=al_scorer)
                    else:
                        sampler_config = SamplerConfig(n_samples=n_samples)
                        labeled_set, added_set = agent.query(sampler_config=sampler_config)

                    self.assertEqual(n_old_labeled + len(added_set), len(labeled_set))
                    assert set(added_set).issubset(labeled_set)
                    self.assertEqual(len(list(set(agent.labeled_set) & set(agent.unlabeled_set))), 0)
                    self.assertEqual(n_old_unlabeled - len(added_set), len(agent.unlabeled_set))

    def test_agent_wrong_scores(self):
        self.api_workflow_client.embedding_id = "embedding_id_xyz"

        agent = ActiveLearningAgent(self.api_workflow_client, preselected_tag_name="preselected_tag_name_xyz")
        method = SamplingMethod.CORAL
        n_samples = len(agent.labeled_set) + 2

        n_predictions = len(agent.unlabeled_set) - 3  # the -3 should cause en error
        predictions = np.random.rand(n_predictions, 10).astype(np.float32)
        predictions_normalized = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
        al_scorer = ScorerClassification(predictions_normalized)

        sampler_config = SamplerConfig(n_samples=n_samples, method=method)
        with self.assertRaises(ValueError):
            labeled_set, added_set = agent.query(sampler_config=sampler_config, al_scorer=al_scorer)
