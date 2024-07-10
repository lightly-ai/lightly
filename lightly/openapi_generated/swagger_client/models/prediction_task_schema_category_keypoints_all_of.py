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
    from pydantic.v1 import BaseModel, Field, conint, conlist, constr
except ImportError:
    # Pydantic v1
    from pydantic import BaseModel, Field, conint, conlist, constr

class PredictionTaskSchemaCategoryKeypointsAllOf(BaseModel):
    """
    The link between the categoryId and the name that should be used
    """
    keypoint_names: Optional[conlist(constr(strict=True, min_length=1))] = Field(None, alias="keypointNames", description="The names of the individual keypoints. E.g left-shoulder, right-shoulder, nose, etc. Must be of equal length as the number of keypoints of a keypoint detection. ")
    keypoint_skeleton: Optional[conlist(conlist(conint(strict=True, ge=0), max_items=2, min_items=2))] = Field(None, alias="keypointSkeleton", description="The keypoint skeleton of a category. It is used to show the overall connectivity between keypoints. Each entry in the array describes a a single connection between two keypoints by their index. e.g [1,3],[2,4],[3,4] ")
    __properties = ["keypointNames", "keypointSkeleton"]

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
    def from_json(cls, json_str: str) -> PredictionTaskSchemaCategoryKeypointsAllOf:
        """Create an instance of PredictionTaskSchemaCategoryKeypointsAllOf from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> PredictionTaskSchemaCategoryKeypointsAllOf:
        """Create an instance of PredictionTaskSchemaCategoryKeypointsAllOf from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return PredictionTaskSchemaCategoryKeypointsAllOf.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in PredictionTaskSchemaCategoryKeypointsAllOf) in the input: " + str(obj))

        _obj = PredictionTaskSchemaCategoryKeypointsAllOf.parse_obj({
            "keypoint_names": obj.get("keypointNames"),
            "keypoint_skeleton": obj.get("keypointSkeleton")
        })
        return _obj

