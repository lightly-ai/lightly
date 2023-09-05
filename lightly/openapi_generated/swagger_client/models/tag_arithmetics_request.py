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


from typing import Optional
from pydantic import Extra, BaseModel, Field, constr, validator
from lightly.openapi_generated.swagger_client.models.tag_arithmetics_operation import (
    TagArithmeticsOperation,
)
from lightly.openapi_generated.swagger_client.models.tag_creator import TagCreator


class TagArithmeticsRequest(BaseModel):
    """
    TagArithmeticsRequest
    """

    tag_id1: constr(strict=True) = Field(
        ..., alias="tagId1", description="MongoDB ObjectId"
    )
    tag_id2: constr(strict=True) = Field(
        ..., alias="tagId2", description="MongoDB ObjectId"
    )
    operation: TagArithmeticsOperation = Field(...)
    new_tag_name: Optional[constr(strict=True, min_length=3)] = Field(
        None, alias="newTagName", description="The name of the tag"
    )
    creator: Optional[TagCreator] = None
    run_id: Optional[constr(strict=True)] = Field(
        None, alias="runId", description="MongoDB ObjectId"
    )
    __properties = ["tagId1", "tagId2", "operation", "newTagName", "creator", "runId"]

    @validator("tag_id1")
    def tag_id1_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator("tag_id2")
    def tag_id2_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator("new_tag_name")
    def new_tag_name_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9 .:;=@_-]+$", value):
            raise ValueError(
                r"must validate the regular expression /^[a-zA-Z0-9][a-zA-Z0-9 .:;=@_-]+$/"
            )
        return value

    @validator("run_id")
    def run_id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
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
    def from_json(cls, json_str: str) -> TagArithmeticsRequest:
        """Create an instance of TagArithmeticsRequest from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias, exclude={}, exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> TagArithmeticsRequest:
        """Create an instance of TagArithmeticsRequest from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return TagArithmeticsRequest.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError(
                    "Error due to additional fields (not defined in TagArithmeticsRequest) in the input: "
                    + str(obj)
                )

        _obj = TagArithmeticsRequest.parse_obj(
            {
                "tag_id1": obj.get("tagId1"),
                "tag_id2": obj.get("tagId2"),
                "operation": obj.get("operation"),
                "new_tag_name": obj.get("newTagName"),
                "creator": obj.get("creator"),
                "run_id": obj.get("runId"),
            }
        )
        return _obj
