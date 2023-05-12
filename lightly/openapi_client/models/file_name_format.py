# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""


from inspect import getfullargspec
import pprint
import re  # noqa: F401
from enum import Enum
from aenum import no_arg  # type: ignore





class FileNameFormat(str, Enum):
    """
    When the filename is output, which format shall be used. E.g for a sample called 'frame0.png' that was uploaded from a datasource 's3://my_bucket/datasets/for_lightly/' in the folder 'car/green/' - NAME: car/green/frame0.png - DATASOURCE_FULL: s3://my_bucket/datasets/for_lightly/car/green/frame0.png - REDIRECTED_READ_URL: https://api.lightly.ai/v1/datasets/{datasetId}/samples/{sampleId}/readurlRedirect?publicToken={jsonWebToken}  
    """

    """
    allowed enum values
    """
    NAME = 'NAME'
    DATASOURCE_FULL = 'DATASOURCE_FULL'
    REDIRECTED_READ_URL = 'REDIRECTED_READ_URL'

