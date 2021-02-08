"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://openapi-generator.tech
"""


import sys
import unittest

import lightly.openapi_generated_with_other_gen.openapi_client
from lightly.openapi_generated_with_other_gen.openapi_client.model.mongo_object_id import MongoObjectID
from lightly.openapi_generated_with_other_gen.openapi_client.model.tag_bit_mask_data import TagBitMaskData
from lightly.openapi_generated_with_other_gen.openapi_client.model.tag_change_data import TagChangeData
from lightly.openapi_generated_with_other_gen.openapi_client.model.tag_name import TagName
globals()['MongoObjectID'] = MongoObjectID
globals()['TagBitMaskData'] = TagBitMaskData
globals()['TagChangeData'] = TagChangeData
globals()['TagName'] = TagName
from lightly.openapi_generated_with_other_gen.openapi_client.model.tag_create_request import TagCreateRequest


class TestTagCreateRequest(unittest.TestCase):
    """TagCreateRequest unit test stubs"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testTagCreateRequest(self):
        """Test TagCreateRequest"""
        # FIXME: construct object with mandatory attributes with example values
        # model = TagCreateRequest()  # noqa: E501
        pass


if __name__ == '__main__':
    unittest.main()
