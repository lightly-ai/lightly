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
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowClient, MockedApiWorkflowSetup


class TestApiWorkflowTags(MockedApiWorkflowSetup):

    def setUp(self) -> None:
        lightly.api.api_workflow_client.__version__ = lightly.__version__
        self.api_workflow_client = MockedApiWorkflowClient(token="token_xyz")

        self.valid_tag_name = self.api_workflow_client._get_all_tags()[0].name
        self.invalid_tag_name = "invalid_tag_name_xyz"
        self.valid_tag_id = self.api_workflow_client._get_all_tags()[0].id
        self.invalid_tag_id = "invalid-tag_id_xyz"
    
    def test_get_tag_name(self):
        self.api_workflow_client.get_tag_and_filenames(tag_name=self.valid_tag_name)
        
    def test_get_tag_name_nonexisting(self):
        with self.assertRaises(ValueError):
            self.api_workflow_client.get_tag_and_filenames(tag_name=self.invalid_tag_name)

    def test_get_tag_id(self):
        self.api_workflow_client.get_tag_and_filenames(tag_id=self.valid_tag_id)

    def test_get_tag_id_nonexisting(self):
        with self.assertRaises(ValueError):
            self.api_workflow_client.get_tag_and_filenames(tag_id=self.invalid_tag_id)

    def test_get_tag_name_id(self):
        self.api_workflow_client.get_tag_and_filenames(tag_name=self.valid_tag_name, tag_id=self.valid_tag_id)

    def test_get_tag_name_id(self):
        self.api_workflow_client.get_tag_and_filenames(tag_name=self.valid_tag_name, tag_id=self.invalid_tag_id)

    def test_get_tag(self):
        with self.assertRaises(ValueError):
            self.api_workflow_client.get_tag_and_filenames()

    def test_get_tag_filenames(self):
        filenames = self.api_workflow_client.download_filenames_from_server()
        self.api_workflow_client.get_tag_and_filenames(tag_id=self.valid_tag_id, filenames_on_server=filenames)

    def test_get_tag_exclude_parent(self):
        self.api_workflow_client.get_tag_and_filenames(tag_id=self.valid_tag_id, exclude_parent_tag=True)








