# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""


import json
import pprint
import re  # noqa: F401
from enum import Enum
from aenum import no_arg  # type: ignore





class SampleType(str, Enum):
    """
    Type of the sample (VideoFrame vs IMAGE vs CROP). Determined by the API!
    """

    """
    allowed enum values
    """
    CROP = 'CROP'
    IMAGE = 'IMAGE'
    VIDEO_FRAME = 'VIDEO_FRAME'

    @classmethod
    def from_json(cls, json_str: str) -> 'SampleType':
        """Create an instance of SampleType from a JSON string"""
        return SampleType(json.loads(json_str))


