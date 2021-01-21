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

class AnnotationData(object):
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
        'id': 'ObjectId',
        'state': 'AnnotationState',
        'dataset_id': 'ObjectId',
        'tag_id': 'ObjectId',
        'partner_id': 'ObjectId',
        'created_at': 'Timestamp',
        'last_modified_at': 'Timestamp',
        'meta': 'AnnotationMetaData',
        'offer': 'AnnotationOfferData'
    }

    attribute_map = {
        'id': '_id',
        'state': 'state',
        'dataset_id': 'datasetId',
        'tag_id': 'tagId',
        'partner_id': 'partnerId',
        'created_at': 'createdAt',
        'last_modified_at': 'lastModifiedAt',
        'meta': 'meta',
        'offer': 'offer'
    }

    def __init__(self, id=None, state=None, dataset_id=None, tag_id=None, partner_id=None, created_at=None, last_modified_at=None, meta=None, offer=None):  # noqa: E501
        """AnnotationData - a model defined in Swagger"""  # noqa: E501
        self._id = None
        self._state = None
        self._dataset_id = None
        self._tag_id = None
        self._partner_id = None
        self._created_at = None
        self._last_modified_at = None
        self._meta = None
        self._offer = None
        self.discriminator = None
        self.id = id
        self.state = state
        self.dataset_id = dataset_id
        self.tag_id = tag_id
        if partner_id is not None:
            self.partner_id = partner_id
        self.created_at = created_at
        self.last_modified_at = last_modified_at
        self.meta = meta
        if offer is not None:
            self.offer = offer

    @property
    def id(self):
        """Gets the id of this AnnotationData.  # noqa: E501


        :return: The id of this AnnotationData.  # noqa: E501
        :rtype: ObjectId
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this AnnotationData.


        :param id: The id of this AnnotationData.  # noqa: E501
        :type: ObjectId
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")  # noqa: E501

        self._id = id

    @property
    def state(self):
        """Gets the state of this AnnotationData.  # noqa: E501


        :return: The state of this AnnotationData.  # noqa: E501
        :rtype: AnnotationState
        """
        return self._state

    @state.setter
    def state(self, state):
        """Sets the state of this AnnotationData.


        :param state: The state of this AnnotationData.  # noqa: E501
        :type: AnnotationState
        """
        if state is None:
            raise ValueError("Invalid value for `state`, must not be `None`")  # noqa: E501

        self._state = state

    @property
    def dataset_id(self):
        """Gets the dataset_id of this AnnotationData.  # noqa: E501


        :return: The dataset_id of this AnnotationData.  # noqa: E501
        :rtype: ObjectId
        """
        return self._dataset_id

    @dataset_id.setter
    def dataset_id(self, dataset_id):
        """Sets the dataset_id of this AnnotationData.


        :param dataset_id: The dataset_id of this AnnotationData.  # noqa: E501
        :type: ObjectId
        """
        if dataset_id is None:
            raise ValueError("Invalid value for `dataset_id`, must not be `None`")  # noqa: E501

        self._dataset_id = dataset_id

    @property
    def tag_id(self):
        """Gets the tag_id of this AnnotationData.  # noqa: E501


        :return: The tag_id of this AnnotationData.  # noqa: E501
        :rtype: ObjectId
        """
        return self._tag_id

    @tag_id.setter
    def tag_id(self, tag_id):
        """Sets the tag_id of this AnnotationData.


        :param tag_id: The tag_id of this AnnotationData.  # noqa: E501
        :type: ObjectId
        """
        if tag_id is None:
            raise ValueError("Invalid value for `tag_id`, must not be `None`")  # noqa: E501

        self._tag_id = tag_id

    @property
    def partner_id(self):
        """Gets the partner_id of this AnnotationData.  # noqa: E501


        :return: The partner_id of this AnnotationData.  # noqa: E501
        :rtype: ObjectId
        """
        return self._partner_id

    @partner_id.setter
    def partner_id(self, partner_id):
        """Sets the partner_id of this AnnotationData.


        :param partner_id: The partner_id of this AnnotationData.  # noqa: E501
        :type: ObjectId
        """

        self._partner_id = partner_id

    @property
    def created_at(self):
        """Gets the created_at of this AnnotationData.  # noqa: E501


        :return: The created_at of this AnnotationData.  # noqa: E501
        :rtype: Timestamp
        """
        return self._created_at

    @created_at.setter
    def created_at(self, created_at):
        """Sets the created_at of this AnnotationData.


        :param created_at: The created_at of this AnnotationData.  # noqa: E501
        :type: Timestamp
        """
        if created_at is None:
            raise ValueError("Invalid value for `created_at`, must not be `None`")  # noqa: E501

        self._created_at = created_at

    @property
    def last_modified_at(self):
        """Gets the last_modified_at of this AnnotationData.  # noqa: E501


        :return: The last_modified_at of this AnnotationData.  # noqa: E501
        :rtype: Timestamp
        """
        return self._last_modified_at

    @last_modified_at.setter
    def last_modified_at(self, last_modified_at):
        """Sets the last_modified_at of this AnnotationData.


        :param last_modified_at: The last_modified_at of this AnnotationData.  # noqa: E501
        :type: Timestamp
        """
        if last_modified_at is None:
            raise ValueError("Invalid value for `last_modified_at`, must not be `None`")  # noqa: E501

        self._last_modified_at = last_modified_at

    @property
    def meta(self):
        """Gets the meta of this AnnotationData.  # noqa: E501


        :return: The meta of this AnnotationData.  # noqa: E501
        :rtype: AnnotationMetaData
        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        """Sets the meta of this AnnotationData.


        :param meta: The meta of this AnnotationData.  # noqa: E501
        :type: AnnotationMetaData
        """
        if meta is None:
            raise ValueError("Invalid value for `meta`, must not be `None`")  # noqa: E501

        self._meta = meta

    @property
    def offer(self):
        """Gets the offer of this AnnotationData.  # noqa: E501


        :return: The offer of this AnnotationData.  # noqa: E501
        :rtype: AnnotationOfferData
        """
        return self._offer

    @offer.setter
    def offer(self, offer):
        """Sets the offer of this AnnotationData.


        :param offer: The offer of this AnnotationData.  # noqa: E501
        :type: AnnotationOfferData
        """

        self._offer = offer

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
        if issubclass(AnnotationData, dict):
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
        if not isinstance(other, AnnotationData):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
