import os
import tempfile
import unittest

import numpy as np

import lightly
from lightly.active_learning.agents.agent import ActiveLearningAgent
from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.active_learning.scorers.classification import ScorerClassification
from lightly.openapi_generated.swagger_client import SamplingMethod
from lightly.openapi_generated.swagger_client.models.tag_data import TagData
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowClient


class TestApiWorkflow(unittest.TestCase):

    def test_reorder_random(self):
        no_random_tries = 100
        for iter in range(no_random_tries):
            numbers_to_choose_from = list(range(100))
            numbers_all = list(np.random.choice(numbers_to_choose_from, 100))
            filenames_on_server = [f"img_{i}" for i in numbers_all]

            api_workflow_client = MockedApiWorkflowClient(host="host_xyz", token="token_xyz",
                                                          dataset_id="dataset_id_xyz")
            api_workflow_client.mappings_api.sample_names = filenames_on_server

            numbers_in_tag = list(np.random.choice(numbers_to_choose_from, 50))
            filenames_for_list = [f"img_{i}" for i in numbers_in_tag]

            list_ordered = api_workflow_client._order_list_by_filenames(filenames_for_list,
                                                                        list_to_order=numbers_in_tag)
            list_desired_order = [i for i in numbers_all if i in numbers_in_tag]
            assert list_ordered == list_desired_order

    def test_reorder_manual(self):
        filenames_on_server = ['a', 'b', 'c']
        api_workflow_client = MockedApiWorkflowClient(host="host_xyz", token="token_xyz",
                                                      dataset_id="dataset_id_xyz")
        api_workflow_client.mappings_api.sample_names = filenames_on_server
        filenames_for_list = ['c', 'a']
        list_to_order = ['cccc', 'aaaa']
        list_ordered = api_workflow_client._order_list_by_filenames(filenames_for_list, list_to_order=list_to_order)
        list_desired_order = ['aaaa', 'cccc']
        assert list_ordered == list_desired_order

    def test_upload_embedding(self, n_data: int = 100):
        # create fake embeddings
        folder_path = tempfile.mkdtemp()
        path_to_embeddings = os.path.join(
            folder_path,
            'embeddings.csv'
        )
        sample_names = [f'img_{i}.jpg' for i in range(n_data)]
        labels = [0] * len(sample_names)
        lightly.utils.save_embeddings(
            path_to_embeddings,
            np.random.randn(n_data, 16),
            labels,
            sample_names
        )

        # Set the workflow with mocked functions
        api_workflow_client = MockedApiWorkflowClient(host="host_xyz", token="token_xyz", dataset_id="dataset_id_xyz")
        # perform the workflow to upload the embeddings
        api_workflow_client.upload_embeddings(path_to_embeddings_csv=path_to_embeddings, name="embedding_xyz")

    def test_sampling(self):
        api_workflow_client = MockedApiWorkflowClient(host="host_xyz", token="token_xyz", dataset_id="dataset_id_xyz")
        api_workflow_client.embedding_id = "embedding_id_xyz"

        sampler_config = SamplerConfig()

        new_tag_data = api_workflow_client.sampling(sampler_config=sampler_config)
        assert isinstance(new_tag_data, TagData)

    def test_agent(self):
        api_workflow_client = MockedApiWorkflowClient(host="host_xyz", token="token_xyz", dataset_id="dataset_id_xyz")
        api_workflow_client.embedding_id = "embedding_id_xyz"

        agent_0 = ActiveLearningAgent(api_workflow_client)
        agent_1 = ActiveLearningAgent(api_workflow_client, query_tag_name="query_tag_name_xyz")
        agent_2 = ActiveLearningAgent(api_workflow_client, query_tag_name="query_tag_name_xyz",
                                    preselected_tag_name="preselected_tag_name_xyz")
        agent_3 = ActiveLearningAgent(api_workflow_client, preselected_tag_name="preselected_tag_name_xyz")

        for agent in [agent_0, agent_1, agent_2, agent_3]:
            for n_samples in [2, 6]:
                sampler_config = SamplerConfig(n_samples=n_samples)
                chosen_filenames = agent.query(sampler_config=sampler_config)

    def test_agent_with_scores(self):
        api_workflow_client = MockedApiWorkflowClient(host="host_xyz", token="token_xyz", dataset_id="dataset_id_xyz")
        api_workflow_client.embedding_id = "embedding_id_xyz"

        agent = ActiveLearningAgent(api_workflow_client, preselected_tag_name="preselected_tag_name_xyz")

        for n_samples in [2, 6]:
            predictions = np.random.rand(len(agent.unlabeled_set), 10)
            predictions_normalized = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
            al_scorer = ScorerClassification(predictions_normalized)
            sampler_config = SamplerConfig(n_samples=n_samples, method=SamplingMethod.CORAL)
            chosen_filenames = agent.query(sampler_config=sampler_config, al_scorer=al_scorer)
