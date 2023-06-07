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


from typing import List
from pydantic import Extra,  BaseModel, Field, conlist, constr, validator
from lightly.openapi_generated.swagger_client.models.prediction_task_schema_category import PredictionTaskSchemaCategory
from lightly.openapi_generated.swagger_client.models.task_type import TaskType

class PredictionTaskSchema(BaseModel):
    """
    The schema for predictions or labels when doing classification, object detection, keypoint detection or instance segmentation 
    """
    name: constr(strict=True, min_length=1) = Field(..., description="A name which is safe to have as a file/folder name in a file system")
    type: TaskType = Field(...)
    categories: conlist(PredictionTaskSchemaCategory) = Field(..., description="An array of the categories that exist for this prediction task. The id needs to be unique")
    __properties = ["name", "type", "categories"]

    @validator('name')
    def name_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9 ._-]+$", value):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9][a-zA-Z0-9 ._-]+$/")
        return value

    class Config:
        """Pydantic configuration"""
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
    def from_json(cls, json_str: str) -> PredictionTaskSchema:
        """Create an instance of PredictionTaskSchema from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in categories (list)
        _items = []
        if self.categories:
            for _item in self.categories:
                if _item:
                    _items.append(_item.to_dict(by_alias=by_alias))
            _dict['categories' if by_alias else 'categories'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> PredictionTaskSchema:
        """Create an instance of PredictionTaskSchema from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return PredictionTaskSchema.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in PredictionTaskSchema) in the input: " + str(obj))

        _obj = PredictionTaskSchema.parse_obj({
            "name": obj.get("name"),
            "type": obj.get("type"),
            "categories": [PredictionTaskSchemaCategory.from_dict(_item) for _item in obj.get("categories")] if obj.get("categories") is not None else None
        })
        return _obj

