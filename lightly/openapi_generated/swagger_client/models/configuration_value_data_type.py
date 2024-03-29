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





class ConfigurationValueDataType(str, Enum):
    """
    We support different data types for the extracted value. This tells Lightly how to interpret the value and also allows you to do different things. - Numeric means the extracted values are in a range and have a lower and upper bound. E.g used for color ranges - Categorical means the extracted values are distinct and can be grouped. This allows us to e.g plot distributions of each unique value within your dataset and to map each unique value to a color    - string: most often used for class/category e.g for city, animal or weather condition   - int: e.g for ratings of a meal   - boolean: for true/false distinctions as e.g isVerified or flashOn   - datetime: e.g for grouping by time   - timestamp: e.g for grouping by time - Other means that the extracted value is important to you but does not fit another category. It is displayed alongside other information in the sample detail. E.g the license 
    """

    """
    allowed enum values
    """
    NUMERIC_INT = 'NUMERIC_INT'
    NUMERIC_FLOAT = 'NUMERIC_FLOAT'
    CATEGORICAL_STRING = 'CATEGORICAL_STRING'
    CATEGORICAL_INT = 'CATEGORICAL_INT'
    CATEGORICAL_BOOLEAN = 'CATEGORICAL_BOOLEAN'
    CATEGORICAL_DATETIME = 'CATEGORICAL_DATETIME'
    CATEGORICAL_TIMESTAMP = 'CATEGORICAL_TIMESTAMP'
    OTHER_STRING = 'OTHER_STRING'

    @classmethod
    def from_json(cls, json_str: str) -> 'ConfigurationValueDataType':
        """Create an instance of ConfigurationValueDataType from a JSON string"""
        return ConfigurationValueDataType(json.loads(json_str))


