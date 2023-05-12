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



from typing import List, Optional
from pydantic import Extra,  BaseModel, Field, StrictInt, conlist, constr, validator
from lightly.openapi_client.models.selection_input_predictions_name import SelectionInputPredictionsName
from lightly.openapi_client.models.selection_input_type import SelectionInputType

class SelectionConfigEntryInput(BaseModel):
    """
    SelectionConfigEntryInput
    """
    type: SelectionInputType = Field(...)
    task: Optional[constr(strict=True)] = Field(None, description="Since we sometimes stitch together SelectionInputTask+ActiveLearningScoreType, they need to follow the same specs of ActiveLearningScoreType. However, this can be an empty string due to internal logic. ")
    score: Optional[constr(strict=True, min_length=1)] = Field(None, description="Type of active learning score")
    key: Optional[constr(strict=True, min_length=1)] = None
    name: Optional[SelectionInputPredictionsName] = None
    dataset_id: Optional[constr(strict=True)] = Field(None, alias="datasetId", description="MongoDB ObjectId")
    tag_name: Optional[constr(strict=True, min_length=3)] = Field(None, alias="tagName", description="The name of the tag")
    random_seed: Optional[StrictInt] = Field(None, alias="randomSeed")
    categories: Optional[conlist(constr(strict=True, min_length=1), min_items=1, unique_items=True)] = None
    __properties = ["type", "task", "score", "key", "name", "datasetId", "tagName", "randomSeed", "categories"]

    @validator('task')
    def task_validate_regular_expression(cls, v):
        if v is None:
            return v

        if not re.match(r"^[a-zA-Z0-9_+=,.@:\/-]*$", v):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9_+=,.@:\/-]*$/")
        return v

    @validator('score')
    def score_validate_regular_expression(cls, v):
        if v is None:
            return v

        if not re.match(r"^[a-zA-Z0-9_+=,.@:\/-]*$", v):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9_+=,.@:\/-]*$/")
        return v

    @validator('dataset_id')
    def dataset_id_validate_regular_expression(cls, v):
        if v is None:
            return v

        if not re.match(r"^[a-f0-9]{24}$", v):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return v

    @validator('tag_name')
    def tag_name_validate_regular_expression(cls, v):
        if v is None:
            return v

        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9 .:;=@_-]+$", v):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9][a-zA-Z0-9 .:;=@_-]+$/")
        return v

    class Config:
        allow_population_by_field_name = True
        validate_assignment = True
        use_enum_values = True
        extra = Extra.forbid

    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.dict(by_alias=True))

    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> SelectionConfigEntryInput:
        """Create an instance of SelectionConfigEntryInput from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> SelectionConfigEntryInput:
        """Create an instance of SelectionConfigEntryInput from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return SelectionConfigEntryInput.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in SelectionConfigEntryInput) in the input: " + str(obj))

        _obj = SelectionConfigEntryInput.parse_obj({
            "type": obj.get("type"),
            "task": obj.get("task"),
            "score": obj.get("score"),
            "key": obj.get("key"),
            "name": obj.get("name"),
            "dataset_id": obj.get("datasetId"),
            "tag_name": obj.get("tagName"),
            "random_seed": obj.get("randomSeed"),
            "categories": obj.get("categories")
        })
        return _obj

