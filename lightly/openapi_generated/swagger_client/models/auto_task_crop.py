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


from typing import List, Union
try:
    # Pydantic >=v1.10.17
    from pydantic.v1 import BaseModel, Field, confloat, conint, conlist
except ImportError:
    # Pydantic v1
    from pydantic import BaseModel, Field, confloat, conint, conlist
from lightly.openapi_generated.swagger_client.models.auto_task_base import AutoTaskBase

class AutoTaskCrop(AutoTaskBase):
    """
    Create a prediction which crops and focuses on a specific part of the image 
    """
    bboxs: conlist(conlist(Union[confloat(le=1, ge=0, strict=True), conint(le=1, ge=0, strict=True)], max_items=4, min_items=4), min_items=1) = Field(..., description="The bounding boxes to focus on.")
    __properties = ["type", "name", "bboxs"]

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
    def from_json(cls, json_str: str) -> AutoTaskCrop:
        """Create an instance of AutoTaskCrop from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> AutoTaskCrop:
        """Create an instance of AutoTaskCrop from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return AutoTaskCrop.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in AutoTaskCrop) in the input: " + str(obj))

        _obj = AutoTaskCrop.parse_obj({
            "type": obj.get("type"),
            "name": obj.get("name"),
            "bboxs": obj.get("bboxs")
        })
        return _obj

