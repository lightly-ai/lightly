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


class DatasourceConfigGCS(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'gcs_project_id': 'str',
        'gcs_credentials': 'str'
    }

    attribute_map = {
        'gcs_project_id': 'gcsProjectId',
        'gcs_credentials': 'gcsCredentials'
    }

    def __init__(self, gcs_project_id=None, gcs_credentials=None, _configuration=None):  # noqa: E501
        """DatasourceConfigGCS - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._gcs_project_id = None
        self._gcs_credentials = None
        self.discriminator = None

        self.gcs_project_id = gcs_project_id
        self.gcs_credentials = gcs_credentials

    @property
    def gcs_project_id(self):
        """Gets the gcs_project_id of this DatasourceConfigGCS.  # noqa: E501

        the projectId where you have your bucket configured  # noqa: E501

        :return: The gcs_project_id of this DatasourceConfigGCS.  # noqa: E501
        :rtype: str
        """
        return self._gcs_project_id

    @gcs_project_id.setter
    def gcs_project_id(self, gcs_project_id):
        """Sets the gcs_project_id of this DatasourceConfigGCS.

        the projectId where you have your bucket configured  # noqa: E501

        :param gcs_project_id: The gcs_project_id of this DatasourceConfigGCS.  # noqa: E501
        :type: str
        """
        if self._configuration.client_side_validation and gcs_project_id is None:
            raise ValueError("Invalid value for `gcs_project_id`, must not be `None`")  # noqa: E501

        self._gcs_project_id = gcs_project_id

    @property
    def gcs_credentials(self):
        """Gets the gcs_credentials of this DatasourceConfigGCS.  # noqa: E501

        this is the content of the credentials JSON file stringified which you downloaded from Google Cloud Platform  # noqa: E501

        :return: The gcs_credentials of this DatasourceConfigGCS.  # noqa: E501
        :rtype: str
        """
        return self._gcs_credentials

    @gcs_credentials.setter
    def gcs_credentials(self, gcs_credentials):
        """Sets the gcs_credentials of this DatasourceConfigGCS.

        this is the content of the credentials JSON file stringified which you downloaded from Google Cloud Platform  # noqa: E501

        :param gcs_credentials: The gcs_credentials of this DatasourceConfigGCS.  # noqa: E501
        :type: str
        """
        if self._configuration.client_side_validation and gcs_credentials is None:
            raise ValueError("Invalid value for `gcs_credentials`, must not be `None`")  # noqa: E501

        self._gcs_credentials = gcs_credentials

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
        if issubclass(DatasourceConfigGCS, dict):
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
        if not isinstance(other, DatasourceConfigGCS):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, DatasourceConfigGCS):
            return True

        return self.to_dict() != other.to_dict()
