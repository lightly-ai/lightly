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





class UserType(str, Enum):
    """
    UserType
    """

    """
    allowed enum values
    """
    FREE = 'FREE'
    PROFESSIONAL = 'PROFESSIONAL'
    ENTERPRISE = 'ENTERPRISE'
    ADMIN = 'ADMIN'

    @classmethod
    def from_json(cls, json_str: str) -> 'UserType':
        """Create an instance of UserType from a JSON string"""
        return UserType(json.loads(json_str))


