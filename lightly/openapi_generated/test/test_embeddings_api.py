# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    OpenAPI spec version: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

from __future__ import absolute_import

import unittest

import swagger_client
from swagger_client.api.embeddings_api import EmbeddingsApi  # noqa: E501
from swagger_client.rest import ApiException


class TestEmbeddingsApi(unittest.TestCase):
    """EmbeddingsApi unit test stubs"""

    def setUp(self):
        self.api = EmbeddingsApi()  # noqa: E501

    def tearDown(self):
        pass

    def test_get_embeddings_by_sample_id(self):
        """Test case for get_embeddings_by_sample_id

        Get all embeddings of a datasets sample  # noqa: E501
        """
        pass

    def test_get_embeddings_csv_write_url_by_id(self):
        """Test case for get_embeddings_csv_write_url_by_id

        Get the signed url to upload an CSVembedding to for a specific dataset  # noqa: E501
        """
        pass


if __name__ == '__main__':
    unittest.main()
