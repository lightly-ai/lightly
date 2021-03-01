# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    OpenAPI spec version: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


import pprint
import re  # noqa: F401

import six

from lightly.openapi_generated.swagger_client.configuration import Configuration


class ApiErrorCode(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    allowed enum values
    """
    BAD_REQUEST = "BAD_REQUEST"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    FORBIDDEN = "FORBIDDEN"
    UNAUTHORIZED = "UNAUTHORIZED"
    NOT_FOUND = "NOT_FOUND"
    MALFORMED_REQUEST = "MALFORMED_REQUEST"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    JOB_CREATION_FAILED = "JOB_CREATION_FAILED"
    USER_NOT_KNOWN = "USER_NOT_KNOWN"
    USER_ACCOUNT_DEACTIVATED = "USER_ACCOUNT_DEACTIVATED"
    USER_ACCOUNT_BLOCKED = "USER_ACCOUNT_BLOCKED"
    TEAM_ACCOUNT_PLAN_INSUFFICIENT = "TEAM_ACCOUNT_PLAN_INSUFFICIENT"
    DATASET_UNKNOWN = "DATASET_UNKNOWN"
    DATASET_TAG_INVALID = "DATASET_TAG_INVALID"
    DATASET_NAME_EXISTS = "DATASET_NAME_EXISTS"
    DATASET_AT_MAX_CAPACITY = "DATASET_AT_MAX_CAPACITY"
    EMBEDDING_UNKNOWN = "EMBEDDING_UNKNOWN"
    EMBEDDING_NAME_EXISTS = "EMBEDDING_NAME_EXISTS"
    EMBEDDING_INVALID = "EMBEDDING_INVALID"
    TAG_UNKNOWN = "TAG_UNKNOWN"
<<<<<<< HEAD
    TAG_NAME_EXISTS = "TAG_NAME_EXISTS"
    TAG_INITIAL_EXISTS = "TAG_INITIAL_EXISTS"
    TAG_PREVTAG_NOT_OF_DATASET = "TAG_PREVTAG_NOT_OF_DATASET"
    SAMPLE_UNKNOWN = "SAMPLE_UNKNOWN"
    SAMPLE_THUMBNAME_UNKNOWN = "SAMPLE_THUMBNAME_UNKNOWN"
    SCORE_UNKNOWN = "SCORE_UNKNOWN"
=======
    TAG_INITIAL_EXISTS = "TAG_INITIAL_EXISTS"
    TAG_PREVTAG_NOT_OF_DATASET = "TAG_PREVTAG_NOT_OF_DATASET"
>>>>>>> aaec1d1... Openapi generated client: v3 on develop_active_learning_branch (#129)

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
    }

    attribute_map = {
    }

    def __init__(self, _configuration=None):  # noqa: E501
        """ApiErrorCode - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration
        self.discriminator = None

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(ApiErrorCode, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, ApiErrorCode):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, ApiErrorCode):
            return True

        return self.to_dict() != other.to_dict()
