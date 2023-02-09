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


class PredictionSingletonBase(object):
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
        'type': 'str',
        'task_name': 'TaskName',
        'crop_dataset_id': 'MongoObjectID',
        'crop_sample_id': 'MongoObjectID',
        'category_id': 'CategoryId',
        'score': 'Score'
    }

    attribute_map = {
        'type': 'type',
        'task_name': 'taskName',
        'crop_dataset_id': 'cropDatasetId',
        'crop_sample_id': 'cropSampleId',
        'category_id': 'categoryId',
        'score': 'score'
    }

    discriminator_value_class_map = {
        'PredictionSingletonObjectDetection': 'PredictionSingletonObjectDetection',
        'PredictionSingletonClassification': 'PredictionSingletonClassification',
        'PredictionSingletonSemanticSegmentation': 'PredictionSingletonSemanticSegmentation',
        'PredictionSingletonInstanceSegmentation': 'PredictionSingletonInstanceSegmentation',
        'PredictionSingletonKeypointDetection': 'PredictionSingletonKeypointDetection'
    }

    def __init__(self, type=None, task_name=None, crop_dataset_id=None, crop_sample_id=None, category_id=None, score=None, _configuration=None):  # noqa: E501
        """PredictionSingletonBase - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._type = None
        self._task_name = None
        self._crop_dataset_id = None
        self._crop_sample_id = None
        self._category_id = None
        self._score = None
        self.discriminator = 'Discriminator{propertyName&#x3D;&#39;type&#39;, mapping&#x3D;null, extensions&#x3D;null}'

        self.type = type
        self.task_name = task_name
        if crop_dataset_id is not None:
            self.crop_dataset_id = crop_dataset_id
        if crop_sample_id is not None:
            self.crop_sample_id = crop_sample_id
        self.category_id = category_id
        self.score = score

    @property
    def type(self):
        """Gets the type of this PredictionSingletonBase.  # noqa: E501


        :return: The type of this PredictionSingletonBase.  # noqa: E501
        :rtype: str
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the type of this PredictionSingletonBase.


        :param type: The type of this PredictionSingletonBase.  # noqa: E501
        :type: str
        """
        if self._configuration.client_side_validation and type is None:
            raise ValueError("Invalid value for `type`, must not be `None`")  # noqa: E501

        self._type = type

    @property
    def task_name(self):
        """Gets the task_name of this PredictionSingletonBase.  # noqa: E501


        :return: The task_name of this PredictionSingletonBase.  # noqa: E501
        :rtype: TaskName
        """
        return self._task_name

    @task_name.setter
    def task_name(self, task_name):
        """Sets the task_name of this PredictionSingletonBase.


        :param task_name: The task_name of this PredictionSingletonBase.  # noqa: E501
        :type: TaskName
        """
        if self._configuration.client_side_validation and task_name is None:
            raise ValueError("Invalid value for `task_name`, must not be `None`")  # noqa: E501

        self._task_name = task_name

    @property
    def crop_dataset_id(self):
        """Gets the crop_dataset_id of this PredictionSingletonBase.  # noqa: E501


        :return: The crop_dataset_id of this PredictionSingletonBase.  # noqa: E501
        :rtype: MongoObjectID
        """
        return self._crop_dataset_id

    @crop_dataset_id.setter
    def crop_dataset_id(self, crop_dataset_id):
        """Sets the crop_dataset_id of this PredictionSingletonBase.


        :param crop_dataset_id: The crop_dataset_id of this PredictionSingletonBase.  # noqa: E501
        :type: MongoObjectID
        """

        self._crop_dataset_id = crop_dataset_id

    @property
    def crop_sample_id(self):
        """Gets the crop_sample_id of this PredictionSingletonBase.  # noqa: E501


        :return: The crop_sample_id of this PredictionSingletonBase.  # noqa: E501
        :rtype: MongoObjectID
        """
        return self._crop_sample_id

    @crop_sample_id.setter
    def crop_sample_id(self, crop_sample_id):
        """Sets the crop_sample_id of this PredictionSingletonBase.


        :param crop_sample_id: The crop_sample_id of this PredictionSingletonBase.  # noqa: E501
        :type: MongoObjectID
        """

        self._crop_sample_id = crop_sample_id

    @property
    def category_id(self):
        """Gets the category_id of this PredictionSingletonBase.  # noqa: E501


        :return: The category_id of this PredictionSingletonBase.  # noqa: E501
        :rtype: CategoryId
        """
        return self._category_id

    @category_id.setter
    def category_id(self, category_id):
        """Sets the category_id of this PredictionSingletonBase.


        :param category_id: The category_id of this PredictionSingletonBase.  # noqa: E501
        :type: CategoryId
        """
        if self._configuration.client_side_validation and category_id is None:
            raise ValueError("Invalid value for `category_id`, must not be `None`")  # noqa: E501

        self._category_id = category_id

    @property
    def score(self):
        """Gets the score of this PredictionSingletonBase.  # noqa: E501


        :return: The score of this PredictionSingletonBase.  # noqa: E501
        :rtype: Score
        """
        return self._score

    @score.setter
    def score(self, score):
        """Sets the score of this PredictionSingletonBase.


        :param score: The score of this PredictionSingletonBase.  # noqa: E501
        :type: Score
        """
        if self._configuration.client_side_validation and score is None:
            raise ValueError("Invalid value for `score`, must not be `None`")  # noqa: E501

        self._score = score

    def get_real_child_model(self, data):
        """Returns the real base class specified by the discriminator"""
        discriminator_value = data[self.discriminator].lower()
        return self.discriminator_value_class_map.get(discriminator_value)

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
        if issubclass(PredictionSingletonBase, dict):
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
        if not isinstance(other, PredictionSingletonBase):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, PredictionSingletonBase):
            return True

        return self.to_dict() != other.to_dict()
