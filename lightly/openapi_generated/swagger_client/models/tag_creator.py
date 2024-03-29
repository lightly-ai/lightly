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





class TagCreator(str, Enum):
    """
    TagCreator
    """

    """
    allowed enum values
    """
    UNKNOWN = 'UNKNOWN'
    USER_WEBAPP = 'USER_WEBAPP'
    USER_PIP = 'USER_PIP'
    USER_PIP_LIGHTLY_MAGIC = 'USER_PIP_LIGHTLY_MAGIC'
    USER_WORKER = 'USER_WORKER'
    SAMPLER_ACTIVE_LEARNING = 'SAMPLER_ACTIVE_LEARNING'
    SAMPLER_CORAL = 'SAMPLER_CORAL'
    SAMPLER_CORESET = 'SAMPLER_CORESET'
    SAMPLER_RANDOM = 'SAMPLER_RANDOM'

    @classmethod
    def from_json(cls, json_str: str) -> 'TagCreator':
        """Create an instance of TagCreator from a JSON string"""
        return TagCreator(json.loads(json_str))


