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



from typing import Optional
from pydantic import Extra,  BaseModel, Field, constr, validator
from lightly.openapi_client.models.image_type import ImageType
from lightly.openapi_client.models.tag_creator import TagCreator

class InitialTagCreateRequest(BaseModel):
    """
    InitialTagCreateRequest
    """
    name: Optional[constr(strict=True, min_length=3)] = Field(None, description="The name of the tag")
    creator: Optional[TagCreator] = None
    img_type: ImageType = Field(..., alias="imgType")
    run_id: Optional[constr(strict=True)] = Field(None, alias="runId", description="MongoDB ObjectId")
    __properties = ["name", "creator", "imgType", "runId"]

    @validator('name')
    def name_validate_regular_expression(cls, v):
        if v is None:
            return v

        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9 .:;=@_-]+$", v):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9][a-zA-Z0-9 .:;=@_-]+$/")
        return v

    @validator('run_id')
    def run_id_validate_regular_expression(cls, v):
        if v is None:
            return v

        if not re.match(r"^[a-f0-9]{24}$", v):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
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
    def from_json(cls, json_str: str) -> InitialTagCreateRequest:
        """Create an instance of InitialTagCreateRequest from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> InitialTagCreateRequest:
        """Create an instance of InitialTagCreateRequest from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return InitialTagCreateRequest.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in InitialTagCreateRequest) in the input: " + str(obj))

        _obj = InitialTagCreateRequest.parse_obj({
            "name": obj.get("name"),
            "creator": obj.get("creator"),
            "img_type": obj.get("imgType"),
            "run_id": obj.get("runId")
        })
        return _obj


