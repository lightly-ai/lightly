import numpy as np

from lightly.active_learning.agents.agent import ActiveLearningAgent
from lightly.active_learning.config.selection_config import SelectionConfig
from lightly.active_learning.scorers import ScorerSemanticSegmentation
from lightly.active_learning.scorers.classification import ScorerClassification
from lightly.openapi_generated.swagger_client.model.sampling_method import \
    SamplingMethod
from tests.api_workflow.mocked_api_workflow_client import \
    MockedApiWorkflowSetup, get_random_MongoObjectID


class TestActiveLearningAgent(MockedApiWorkflowSetup):
    def test_agent(self):
        self.api_workflow_client.embedding_id = get_random_MongoObjectID()

        agent_0 = ActiveLearningAgent(self.api_workflow_client)
        agent_1 = ActiveLearningAgent(self.api_workflow_client, query_tag_name="query_tag_name_xyz")
        agent_2 = ActiveLearningAgent(self.api_workflow_client, query_tag_name="query_tag_name_xyz",
                                      preselected_tag_name="preselected_tag_name_xyz")
        agent_3 = ActiveLearningAgent(self.api_workflow_client, preselected_tag_name="preselected_tag_name_xyz")

        for method in [SamplingMethod.CORAL, SamplingMethod.CORESET, SamplingMethod.RANDOM]:
            for agent in [agent_0, agent_1, agent_2, agent_3]:
                for batch_size in [2, 6]:
                    with self.subTest(
                            method=method,
                            agent=agent.__dict__,
                            batch_size=batch_size
                    ):
                        n_old_labeled = len(agent.labeled_set)
                        n_old_unlabeled = len(agent.unlabeled_set)

                        n_samples = len(agent.labeled_set) + batch_size
                        if method == SamplingMethod.CORAL and len(agent.labeled_set) == 0:
                            selection_config = SelectionConfig(n_samples=n_samples, method=SamplingMethod.CORESET)
                        else:
                            selection_config = SelectionConfig(n_samples=n_samples, method=method)

                        if selection_config.method == SamplingMethod.CORAL:
                            predictions = np.random.rand(len(agent.query_set), 10).astype(np.float32)
                            predictions_normalized = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
                            al_scorer = ScorerClassification(predictions_normalized)
                            agent.query(selection_config=selection_config, al_scorer=al_scorer)
                        else:
                            selection_config = SelectionConfig(n_samples=n_samples)
                            agent.query(selection_config=selection_config)

                        labeled_set, added_set = agent.labeled_set, agent.added_set

                        self.assertEqual(n_old_labeled + len(added_set), len(labeled_set))
                        self.assertTrue(set(added_set).issubset(labeled_set))
                        self.assertEqual(len(list(set(agent.labeled_set) & set(agent.unlabeled_set))), 0)
                        self.assertEqual(n_old_unlabeled - len(added_set), len(agent.unlabeled_set))

    def test_agent_wrong_number_of_scores(self):
        self.api_workflow_client.embedding_id = "embedding_id_xyz"

        agent = ActiveLearningAgent(self.api_workflow_client, preselected_tag_name="preselected_tag_name_xyz")
        method = SamplingMethod.CORAL
        n_samples = len(agent.labeled_set) + 2

        n_predictions = len(agent.query_set) - 3  # the -3 should cause an error
        predictions = np.random.rand(n_predictions, 10).astype(np.float32)
        predictions_normalized = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
        al_scorer = ScorerClassification(predictions_normalized)

        selection_config = SelectionConfig(n_samples=n_samples, method=method)
        with self.assertRaises(ValueError):
            agent.query(selection_config=selection_config, al_scorer=al_scorer)

    def test_agent_with_generator(self):
        self.api_workflow_client.embedding_id = "embedding_id_xyz"
        width = 32
        height = 32
        no_classes = 13

        agent = ActiveLearningAgent(self.api_workflow_client, preselected_tag_name="preselected_tag_name_xyz")
        method = SamplingMethod.CORAL
        n_samples = len(agent.labeled_set) + 2

        n_predictions = len(agent.query_set)
        predictions = np.random.rand(n_predictions, no_classes, width, height).astype(np.float32)
        predictions_normalized = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
        predictions_generator = (predictions_normalized[i] for i in range(n_predictions))
        al_scorer = ScorerSemanticSegmentation(predictions_generator)

        selection_config = SelectionConfig(n_samples=n_samples, method=method)
        agent.query(selection_config=selection_config, al_scorer=al_scorer)

        # make sure we throw an error if generator is already consumed
        with self.assertRaises(ValueError):
            agent.upload_scores(al_scorer)

    def test_agent_added_set_before_query(self):

        self.api_workflow_client.embedding_id = "embedding_id_xyz"
        agent = ActiveLearningAgent(
            self.api_workflow_client,
            preselected_tag_name="preselected_tag_name_xyz"
        )

        agent.query_set
        agent.labeled_set
        agent.unlabeled_set
        with self.assertRaises(RuntimeError):
            agent.added_set

    def test_agent_query_too_few(self):

        self.api_workflow_client.embedding_id = "embedding_id_xyz"
        agent = ActiveLearningAgent(
            self.api_workflow_client,
            preselected_tag_name="preselected_tag_name_xyz",
        )

        # sample 0 samples
        selection_config = SelectionConfig(
            n_samples=0,
            method=SamplingMethod.RANDOM
        )

        agent.query(selection_config)

    def test_agent_only_upload_scores(self):
        self.api_workflow_client.embedding_id = "embedding_id_xyz"
        agent = ActiveLearningAgent(
            self.api_workflow_client,
            preselected_tag_name="preselected_tag_name_xyz",
        )

        n_predictions = len(agent.query_set)
        predictions = np.random.rand(n_predictions, 10).astype(np.float32)
        predictions_normalized = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
        al_scorer = ScorerClassification(predictions_normalized)

        agent.upload_scores(al_scorer)

    def test_agent_without_embedding_id(self):
        agent = ActiveLearningAgent(
            self.api_workflow_client,
            preselected_tag_name="preselected_tag_name_xyz"
        )
        method = SamplingMethod.CORAL
        n_samples = len(agent.labeled_set) + 2

        n_predictions = len(agent.query_set)
        predictions = np.random.rand(n_predictions, 10).astype(np.float32)
        predictions_normalized = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
        al_scorer = ScorerClassification(predictions_normalized)

        selection_config = SelectionConfig(n_samples=n_samples, method=method)
        agent.query(selection_config=selection_config, al_scorer=al_scorer)



