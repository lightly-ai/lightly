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

class TagCreateRequest(object):
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
        'name': 'TagName',
        'prev_tag_id': 'MongoObjectID',
        'bit_mask_data': 'TagBitMaskData',
        'tot_size': 'int',
        'changes': 'TagChangeData'
    }

    attribute_map = {
        'name': 'name',
        'prev_tag_id': 'prevTagId',
        'bit_mask_data': 'bitMaskData',
        'tot_size': 'totSize',
        'changes': 'changes'
    }

    def __init__(self, name=None, prev_tag_id=None, bit_mask_data=None, tot_size=None, changes=None):  # noqa: E501
        """TagCreateRequest - a model defined in Swagger"""  # noqa: E501
        self._name = None
        self._prev_tag_id = None
        self._bit_mask_data = None
        self._tot_size = None
        self._changes = None
        self.discriminator = None
        self.name = name
        self.prev_tag_id = prev_tag_id
        self.bit_mask_data = bit_mask_data
        self.tot_size = tot_size
        if changes is not None:
            self.changes = changes

    @property
    def name(self):
        """Gets the name of this TagCreateRequest.  # noqa: E501


        :return: The name of this TagCreateRequest.  # noqa: E501
        :rtype: TagName
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this TagCreateRequest.


        :param name: The name of this TagCreateRequest.  # noqa: E501
        :type: TagName
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")  # noqa: E501

        self._name = name

    @property
    def prev_tag_id(self):
        """Gets the prev_tag_id of this TagCreateRequest.  # noqa: E501


        :return: The prev_tag_id of this TagCreateRequest.  # noqa: E501
        :rtype: MongoObjectID
        """
        return self._prev_tag_id

    @prev_tag_id.setter
    def prev_tag_id(self, prev_tag_id):
        """Sets the prev_tag_id of this TagCreateRequest.


        :param prev_tag_id: The prev_tag_id of this TagCreateRequest.  # noqa: E501
        :type: MongoObjectID
        """
        if prev_tag_id is None:
            raise ValueError("Invalid value for `prev_tag_id`, must not be `None`")  # noqa: E501

        self._prev_tag_id = prev_tag_id

    @property
    def bit_mask_data(self):
        """Gets the bit_mask_data of this TagCreateRequest.  # noqa: E501


        :return: The bit_mask_data of this TagCreateRequest.  # noqa: E501
        :rtype: TagBitMaskData
        """
        return self._bit_mask_data

    @bit_mask_data.setter
    def bit_mask_data(self, bit_mask_data):
        """Sets the bit_mask_data of this TagCreateRequest.


        :param bit_mask_data: The bit_mask_data of this TagCreateRequest.  # noqa: E501
        :type: TagBitMaskData
        """
        if bit_mask_data is None:
            raise ValueError("Invalid value for `bit_mask_data`, must not be `None`")  # noqa: E501

        self._bit_mask_data = bit_mask_data

    @property
    def tot_size(self):
        """Gets the tot_size of this TagCreateRequest.  # noqa: E501


        :return: The tot_size of this TagCreateRequest.  # noqa: E501
        :rtype: int
        """
        return self._tot_size

    @tot_size.setter
    def tot_size(self, tot_size):
        """Sets the tot_size of this TagCreateRequest.


        :param tot_size: The tot_size of this TagCreateRequest.  # noqa: E501
        :type: int
        """
        if tot_size is None:
            raise ValueError("Invalid value for `tot_size`, must not be `None`")  # noqa: E501

        self._tot_size = tot_size

    @property
    def changes(self):
        """Gets the changes of this TagCreateRequest.  # noqa: E501


        :return: The changes of this TagCreateRequest.  # noqa: E501
        :rtype: TagChangeData
        """
        return self._changes

    @changes.setter
    def changes(self, changes):
        """Sets the changes of this TagCreateRequest.


        :param changes: The changes of this TagCreateRequest.  # noqa: E501
        :type: TagChangeData
        """

        self._changes = changes

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
        if issubclass(TagCreateRequest, dict):
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
        if not isinstance(other, TagCreateRequest):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
