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

            api_workflow_client = MockedApiWorkflowClient(token="token_xyz", dataset_id="dataset_id_xyz")
            api_workflow_client.mappings_api.sample_names = filenames_on_server

            numbers_in_tag = list(np.random.choice(numbers_to_choose_from, 50))
            filenames_for_list = [f"img_{i}" for i in numbers_in_tag]

            list_ordered = api_workflow_client._order_list_by_filenames(filenames_for_list,
                                                                        list_to_order=numbers_in_tag)
            list_desired_order = [i for i in numbers_all if i in numbers_in_tag]
            assert list_ordered == list_desired_order

    def test_reorder_manual(self):
        filenames_on_server = ['a', 'b', 'c']
        api_workflow_client = MockedApiWorkflowClient(token="token_xyz", dataset_id="dataset_id_xyz")
        api_workflow_client.mappings_api.sample_names = filenames_on_server
        filenames_for_list = ['c', 'a']
        list_to_order = ['cccc', 'aaaa']
        list_ordered = api_workflow_client._order_list_by_filenames(filenames_for_list, list_to_order=list_to_order)
        list_desired_order = ['aaaa', 'cccc']
        assert list_ordered == list_desired_order




