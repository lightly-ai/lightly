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


class SampleData(object):
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
        'id': 'MongoObjectID',
        'is_thumbnail': 'bool',
        'thumb_name': 'str',
        'meta': 'SampleMetaData'
    }

    attribute_map = {
        'id': '_id',
        'is_thumbnail': 'isThumbnail',
        'thumb_name': 'thumbName',
        'meta': 'meta'
    }

    def __init__(self, id=None, is_thumbnail=None, thumb_name=None, meta=None, _configuration=None):  # noqa: E501
        """SampleData - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._id = None
        self._is_thumbnail = None
        self._thumb_name = None
        self._meta = None
        self.discriminator = None

        self.id = id
        self.is_thumbnail = is_thumbnail
        self.thumb_name = thumb_name
        self.meta = meta

    @property
    def id(self):
        """Gets the id of this SampleData.  # noqa: E501


        :return: The id of this SampleData.  # noqa: E501
        :rtype: MongoObjectID
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this SampleData.


        :param id: The id of this SampleData.  # noqa: E501
        :type: MongoObjectID
        """
        if self._configuration.client_side_validation and id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def is_thumbnail(self):
        """Gets the is_thumbnail of this SampleData.  # noqa: E501


        :return: The is_thumbnail of this SampleData.  # noqa: E501
        :rtype: bool
        """
        return self._is_thumbnail

    @is_thumbnail.setter
    def is_thumbnail(self, is_thumbnail):
        """Sets the is_thumbnail of this SampleData.


        :param is_thumbnail: The is_thumbnail of this SampleData.  # noqa: E501
        :type: bool
        """
        if self._configuration.client_side_validation and is_thumbnail is None:
            raise ValueError("Invalid value for `is_thumbnail`, must not be `None`")  # noqa: E501

        self._is_thumbnail = is_thumbnail

    @property
    def thumb_name(self):
        """Gets the thumb_name of this SampleData.  # noqa: E501


        :return: The thumb_name of this SampleData.  # noqa: E501
        :rtype: str
        """
        return self._thumb_name

    @thumb_name.setter
    def thumb_name(self, thumb_name):
        """Sets the thumb_name of this SampleData.


        :param thumb_name: The thumb_name of this SampleData.  # noqa: E501
        :type: str
        """
        if self._configuration.client_side_validation and thumb_name is None:
            raise ValueError("Invalid value for `thumb_name`, must not be `None`")  # noqa: E501

        self._thumb_name = thumb_name

    @property
    def meta(self):
        """Gets the meta of this SampleData.  # noqa: E501


        :return: The meta of this SampleData.  # noqa: E501
        :rtype: SampleMetaData
        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        """Sets the meta of this SampleData.


        :param meta: The meta of this SampleData.  # noqa: E501
        :type: SampleMetaData
        """
        if self._configuration.client_side_validation and meta is None:
            raise ValueError("Invalid value for `meta`, must not be `None`")  # noqa: E501

        self._meta = meta

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
        if issubclass(SampleData, dict):
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
        if not isinstance(other, SampleData):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, SampleData):
            return True

        return self.to_dict() != other.to_dict()
