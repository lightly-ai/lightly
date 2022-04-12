import os
import tempfile
import unittest
import warnings

import numpy as np

import lightly
from lightly.active_learning.agents.agent import ActiveLearningAgent
from lightly.active_learning.config.selection_config import SelectionConfig
from lightly.active_learning.scorers.classification import ScorerClassification
from lightly.openapi_generated.swagger_client import SamplingMethod
from lightly.openapi_generated.swagger_client.model.tag_data import TagData
from tests.api_workflow.mocked_api_workflow_client import MockedApiWorkflowClient, MockedApiWorkflowSetup


class TestApiWorkflowTags(MockedApiWorkflowSetup):

    def setUp(self) -> None:
        lightly.api.api_workflow_client.__version__ = lightly.__version__
        warnings.filterwarnings("ignore", category=UserWarning)
        self.api_workflow_client = MockedApiWorkflowClient(token="token_xyz")

        self.valid_tag_name = self.api_workflow_client.get_all_tags()[0].name
        self.invalid_tag_name = "invalid_tag_name_xyz"
        self.valid_tag_id = self.api_workflow_client.get_all_tags()[0].id
        self.invalid_tag_id = "invalid-tag_id_xyz"

    def tearDown(self) -> None:
        warnings.resetwarnings()

    def test_get_all_tags(self):
        self.api_workflow_client.get_all_tags()
    
    def test_get_tag_name(self):
        self.api_workflow_client.get_tag_by_name(tag_name=self.valid_tag_name)
        
    def test_get_tag_name_nonexisting(self):
        with self.assertRaises(ValueError):
            self.api_workflow_client.get_tag_by_name(tag_name=self.invalid_tag_name)

    def test_get_tag_id(self):
        self.api_workflow_client.get_tag_by_id(tag_id=self.valid_tag_id)

    def test_get_filenames_in_tag(self):
        tag_data = self.api_workflow_client.get_tag_by_name(tag_name=self.valid_tag_name)
        self.api_workflow_client.get_filenames_in_tag(tag_data)

    def test_get_filenames_in_tag_with_filenames(self):
        tag_data = self.api_workflow_client.get_tag_by_name(tag_name=self.valid_tag_name)
        filenames = self.api_workflow_client.get_filenames()
        self.api_workflow_client.get_filenames_in_tag(tag_data, filenames)

    def test_get_filenames_in_tag_exclude_parent(self):
        tag_data = self.api_workflow_client.get_tag_by_name(tag_name=self.valid_tag_name)
        self.api_workflow_client.get_filenames_in_tag(tag_data, exclude_parent_tag=True)

    def test_get_filenames_in_tag_with_filenames_exclude_parent(self):
        tag_data = self.api_workflow_client.get_tag_by_name(tag_name=self.valid_tag_name)
        filenames = self.api_workflow_client.get_filenames()
        self.api_workflow_client.get_filenames_in_tag(tag_data, filenames, exclude_parent_tag=True)

    def test_create_tag_from_filenames(self):
        filenames_server = self.api_workflow_client.get_filenames()
        filenames_new_tag = filenames_server[:10][::3]
        self.api_workflow_client.create_tag_from_filenames(filenames_new_tag, new_tag_name="funny_new_tag")

    def test_create_tag_from_filenames(self):
        filenames_server = self.api_workflow_client.get_filenames()
        filenames_new_tag = filenames_server[:10][::3]
        filenames_new_tag[0] = 'some-random-non-existing-filename.jpg'
        with self.assertRaises(RuntimeError):
            self.api_workflow_client.create_tag_from_filenames(filenames_new_tag, new_tag_name="funny_new_tag")

    def test_delete_tag_by_id(self):
        self.api_workflow_client.delete_tag_by_id(self.valid_tag_id)

