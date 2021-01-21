# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://openapi-generator.tech
"""


from __future__ import absolute_import

import unittest

import lightly.openapi_generated.openapi_client
from lightly.openapi_generated.openapi_client.api.tags_api import TagsApi  # noqa: E501
from lightly.openapi_generated.openapi_client.rest import ApiException


class TestTagsApi(unittest.TestCase):
    """TagsApi unit test stubs"""

    def setUp(self):
        self.api = openapi_client.api.tags_api.TagsApi()  # noqa: E501

    def tearDown(self):
        pass

    def test_create_tag_by_dataset_id(self):
        """Test case for create_tag_by_dataset_id

        create new tag for dataset  # noqa: E501
        """
        pass

    def test_get_tags_by_dataset_id(self):
        """Test case for get_tags_by_dataset_id

        Get all tags of a dataset  # noqa: E501
        """
        pass

    def test_trigger_sampling_by_id(self):
        """Test case for trigger_sampling_by_id

        Trigger a sampling on a specific tag of a dataset with specific prior uploaded csv embedding  # noqa: E501
        """
        pass


if __name__ == '__main__':
    unittest.main()
