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
from inspect import getfullargspec
import pprint
import re  # noqa: F401
import json

from typing_extensions import Annotated




from pydantic import Extra,  BaseModel, Field, conint, constr, validator

class TagActiveLearningScoresData(BaseModel):
    """
    Array of scores belonging to tag
    """
    id: constr(strict=True) = Field(..., description="MongoDB ObjectId")
    tag_id: constr(strict=True) = Field(..., alias="tagId", description="MongoDB ObjectId")
    score_type: constr(strict=True, min_length=1) = Field(..., alias="scoreType", description="Type of active learning score")
    created_at: conint(strict=True, ge=0) = Field(..., alias="createdAt", description="unix timestamp in milliseconds")
    __properties = ["id", "tagId", "scoreType", "createdAt"]

    @validator('id')
    def id_validate_regular_expression(cls, v):
        if not re.match(r"^[a-f0-9]{24}$", v):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return v

    @validator('tag_id')
    def tag_id_validate_regular_expression(cls, v):
        if not re.match(r"^[a-f0-9]{24}$", v):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return v

    @validator('score_type')
    def score_type_validate_regular_expression(cls, v):
        if not re.match(r"^[a-zA-Z0-9_+=,.@:\/-]*$", v):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9_+=,.@:\/-]*$/")
        return v

    class Config:
        allow_population_by_field_name = True
        validate_assignment = True
        use_enum_values = True
        extra = Extra.forbid

    def to_str(self, by_alias: bool = False) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.dict(by_alias=by_alias))

    def to_json(self, by_alias: bool = False) -> str:
        """Returns the JSON representation of the model"""
        return json.dumps(self.to_dict(by_alias=by_alias))

    @classmethod
    def from_json(cls, json_str: str) -> TagActiveLearningScoresData:
        """Create an instance of TagActiveLearningScoresData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> TagActiveLearningScoresData:
        """Create an instance of TagActiveLearningScoresData from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return TagActiveLearningScoresData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in TagActiveLearningScoresData) in the input: " + str(obj))

        _obj = TagActiveLearningScoresData.parse_obj({
            "id": obj.get("id"),
            "tag_id": obj.get("tagId"),
            "score_type": obj.get("scoreType"),
            "created_at": obj.get("createdAt")
        })
        return _obj


