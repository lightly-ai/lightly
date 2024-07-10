# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""


from __future__ import annotations
import pprint
import re  # noqa: F401
import json


from typing import List, Optional
try:
    # Pydantic >=v1.10.17
    from pydantic.v1 import BaseModel, StrictStr, conlist
except ImportError:
    # Pydantic v1
    from pydantic import BaseModel, StrictStr, conlist
from lightly.openapi_generated.swagger_client.models.object_detection_prediction import ObjectDetectionPrediction

class DetectionFramePrediction(BaseModel):
    """
    DetectionFramePrediction
    """
    filename: Optional[StrictStr] = None
    predictions: Optional[conlist(ObjectDetectionPrediction)] = None
    __properties = ["filename", "predictions"]

    class Config:
        """Pydantic configuration"""
        allow_population_by_field_name = True
        validate_assignment = True
        use_enum_values = True
        extra = "forbid"

    def to_str(self, by_alias: bool = False) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.dict(by_alias=by_alias))

    def to_json(self, by_alias: bool = False) -> str:
        """Returns the JSON representation of the model"""
        return json.dumps(self.to_dict(by_alias=by_alias))

    @classmethod
    def from_json(cls, json_str: str) -> DetectionFramePrediction:
        """Create an instance of DetectionFramePrediction from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in predictions (list)
        _items = []
        if self.predictions:
            for _item in self.predictions:
                if _item:
                    _items.append(_item.to_dict(by_alias=by_alias))
            _dict['predictions' if by_alias else 'predictions'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DetectionFramePrediction:
        """Create an instance of DetectionFramePrediction from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return DetectionFramePrediction.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DetectionFramePrediction) in the input: " + str(obj))

        _obj = DetectionFramePrediction.parse_obj({
            "filename": obj.get("filename"),
            "predictions": [ObjectDetectionPrediction.from_dict(_item) for _item in obj.get("predictions")] if obj.get("predictions") is not None else None
        })
        return _obj

