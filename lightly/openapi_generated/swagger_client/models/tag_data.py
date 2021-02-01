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


class TagData(object):
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
        'dataset_id': 'MongoObjectID',
        'prev_tag_id': 'str',
        'name': 'TagName',
        'bit_mask_data': 'TagBitMaskData',
        'tot_size': 'int',
        'created_at': 'Timestamp',
        'changes': 'TagChangeData'
    }

    attribute_map = {
        'id': 'id',
        'dataset_id': 'datasetId',
        'prev_tag_id': 'prevTagId',
        'name': 'name',
        'bit_mask_data': 'bitMaskData',
        'tot_size': 'totSize',
        'created_at': 'createdAt',
        'changes': 'changes'
    }

    def __init__(self, id=None, dataset_id=None, prev_tag=None, name=None, bit_mask_data=None, tot_size=None, created_at=None, changes=None):  # noqa: E501
        """TagData - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._id = None
        self._dataset_id = None
        self._prev_tag_id = None
        self._name = None
        self._bit_mask_data = None
        self._tot_size = None
        self._created_at = None
        self._changes = None
        self.discriminator = None

        self.id = id
        self.dataset_id = dataset_id
        self.prev_tag_id = prev_tag_id
        self.name = name
        self.bit_mask_data = bit_mask_data
        self.tot_size = tot_size
        self.created_at = created_at
        if changes is not None:
            self.changes = changes

    @property
    def id(self):
        """Gets the id of this TagData.  # noqa: E501


        :return: The id of this TagData.  # noqa: E501
        :rtype: MongoObjectID
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this TagData.


        :param id: The id of this TagData.  # noqa: E501
        :type: MongoObjectID
        """
        if self._configuration.client_side_validation and id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def dataset_id(self):
        """Gets the dataset_id of this TagData.  # noqa: E501


        :return: The dataset_id of this TagData.  # noqa: E501
        :rtype: MongoObjectID
        """
        return self._dataset_id

    @dataset_id.setter
    def dataset_id(self, dataset_id):
        """Sets the dataset_id of this TagData.


        :param dataset_id: The dataset_id of this TagData.  # noqa: E501
        :type: MongoObjectID
        """
        if self._configuration.client_side_validation and dataset_id is None:
            raise ValueError("Invalid value for `dataset_id`, must not be `None`")  # noqa: E501

        self._dataset_id = dataset_id

    @property
    def prev_tag_id(self):
        """Gets the prev_tag_id of this TagData.  # noqa: E501

        MongoObjectID or null  # noqa: E501

        :return: The prev_tag_id of this TagData.  # noqa: E501
        :rtype: str
        """
        return self._prev_tag_id

    @prev_tag_id.setter
    def prev_tag_id(self, prev_tag_id):
        """Sets the prev_tag_id of this TagData.

        MongoObjectID or null  # noqa: E501

        :param prev_tag_id: The prev_tag_id of this TagData.  # noqa: E501
        :type: str
        """

        self._prev_tag_id = prev_tag_id

    @property
    def name(self):
        """Gets the name of this TagData.  # noqa: E501


        :return: The name of this TagData.  # noqa: E501
        :rtype: TagName
        """
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this TagData.


        :param name: The name of this TagData.  # noqa: E501
        :type: TagName
        """
        if self._configuration.client_side_validation and name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")  # noqa: E501

        self._name = name

    @property
    def bit_mask_data(self):
        """Gets the bit_mask_data of this TagData.  # noqa: E501


        :return: The bit_mask_data of this TagData.  # noqa: E501
        :rtype: TagBitMaskData
        """
        return self._bit_mask_data

    @bit_mask_data.setter
    def bit_mask_data(self, bit_mask_data):
        """Sets the bit_mask_data of this TagData.


        :param bit_mask_data: The bit_mask_data of this TagData.  # noqa: E501
        :type: TagBitMaskData
        """
        if bit_mask_data is None:
            raise ValueError("Invalid value for `bit_mask_data`, must not be `None`")  # noqa: E501

        self._bit_mask_data = bit_mask_data

    @property
    def tot_size(self):
        """Gets the tot_size of this TagData.  # noqa: E501


        :return: The tot_size of this TagData.  # noqa: E501
        :rtype: int
        """
        return self._tot_size

    @tot_size.setter
    def tot_size(self, tot_size):
        """Sets the tot_size of this TagData.


        :param tot_size: The tot_size of this TagData.  # noqa: E501
        :type: int
        """
        if tot_size is None:
            raise ValueError("Invalid value for `tot_size`, must not be `None`")  # noqa: E501

        self._tot_size = tot_size

    @property
    def created_at(self):
        """Gets the created_at of this TagData.  # noqa: E501


        :return: The created_at of this TagData.  # noqa: E501
        :rtype: Timestamp
        """
        return self._created_at

    @created_at.setter
    def created_at(self, created_at):
        """Sets the created_at of this TagData.


        :param created_at: The created_at of this TagData.  # noqa: E501
        :type: Timestamp
        """
        if self._configuration.client_side_validation and created_at is None:
            raise ValueError("Invalid value for `created_at`, must not be `None`")  # noqa: E501

        self._created_at = created_at

    @property
    def changes(self):
        """Gets the changes of this TagData.  # noqa: E501


        :return: The changes of this TagData.  # noqa: E501
        :rtype: TagChangeData
        """
        return self._changes

    @changes.setter
    def changes(self, changes):
        """Sets the changes of this TagData.


        :param changes: The changes of this TagData.  # noqa: E501
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
        if issubclass(TagData, dict):
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
        if not isinstance(other, TagData):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, TagData):
            return True

        return self.to_dict() != other.to_dict()
