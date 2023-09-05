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


class DatasetCreator(str, Enum):
    """
    DatasetCreator
    """

    """
    allowed enum values
    """
    UNKNOWN = "UNKNOWN"
    USER_WEBAPP = "USER_WEBAPP"
    USER_PIP = "USER_PIP"
    USER_PIP_LIGHTLY_MAGIC = "USER_PIP_LIGHTLY_MAGIC"
    USER_WORKER = "USER_WORKER"

    @classmethod
    def from_json(cls, json_str: str) -> "DatasetCreator":
        """Create an instance of DatasetCreator from a JSON string"""
        return DatasetCreator(json.loads(json_str))
