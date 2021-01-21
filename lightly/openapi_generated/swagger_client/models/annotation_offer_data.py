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

class AnnotationOfferData(object):
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
        'cost': 'float',
        'completed_by': 'Timestamp'
    }

    attribute_map = {
        'cost': 'cost',
        'completed_by': 'completedBy'
    }

    def __init__(self, cost=None, completed_by=None):  # noqa: E501
        """AnnotationOfferData - a model defined in Swagger"""  # noqa: E501
        self._cost = None
        self._completed_by = None
        self.discriminator = None
        if cost is not None:
            self.cost = cost
        if completed_by is not None:
            self.completed_by = completed_by

    @property
    def cost(self):
        """Gets the cost of this AnnotationOfferData.  # noqa: E501


        :return: The cost of this AnnotationOfferData.  # noqa: E501
        :rtype: float
        """
        return self._cost

    @cost.setter
    def cost(self, cost):
        """Sets the cost of this AnnotationOfferData.


        :param cost: The cost of this AnnotationOfferData.  # noqa: E501
        :type: float
        """

        self._cost = cost

    @property
    def completed_by(self):
        """Gets the completed_by of this AnnotationOfferData.  # noqa: E501


        :return: The completed_by of this AnnotationOfferData.  # noqa: E501
        :rtype: Timestamp
        """
        return self._completed_by

    @completed_by.setter
    def completed_by(self, completed_by):
        """Sets the completed_by of this AnnotationOfferData.


        :param completed_by: The completed_by of this AnnotationOfferData.  # noqa: E501
        :type: Timestamp
        """

        self._completed_by = completed_by

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
        if issubclass(AnnotationOfferData, dict):
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
        if not isinstance(other, AnnotationOfferData):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
