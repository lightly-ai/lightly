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
from pydantic import Extra,  BaseModel, Field, StrictInt, constr, validator
from lightly.openapi_client.models.tag_change_data import TagChangeData
from lightly.openapi_client.models.tag_creator import TagCreator

class TagCreateRequest(BaseModel):
    """
    TagCreateRequest
    """
    name: constr(strict=True, min_length=3) = Field(..., description="The name of the tag")
    prev_tag_id: constr(strict=True) = Field(..., alias="prevTagId", description="MongoDB ObjectId")
    query_tag_id: Optional[constr(strict=True)] = Field(None, alias="queryTagId", description="MongoDB ObjectId")
    preselected_tag_id: Optional[constr(strict=True)] = Field(None, alias="preselectedTagId", description="MongoDB ObjectId")
    bit_mask_data: constr(strict=True) = Field(..., alias="bitMaskData", description="BitMask as a base16 (hex) string")
    tot_size: StrictInt = Field(..., alias="totSize")
    creator: Optional[TagCreator] = None
    changes: Optional[TagChangeData] = None
    run_id: Optional[constr(strict=True)] = Field(None, alias="runId", description="MongoDB ObjectId")
    __properties = ["name", "prevTagId", "queryTagId", "preselectedTagId", "bitMaskData", "totSize", "creator", "changes", "runId"]

    @validator('name')
    def name_validate_regular_expression(cls, v):
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9 .:;=@_-]+$", v):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9][a-zA-Z0-9 .:;=@_-]+$/")
        return v

    @validator('prev_tag_id')
    def prev_tag_id_validate_regular_expression(cls, v):
        if not re.match(r"^[a-f0-9]{24}$", v):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return v

    @validator('query_tag_id')
    def query_tag_id_validate_regular_expression(cls, v):
        if v is None:
            return v

        if not re.match(r"^[a-f0-9]{24}$", v):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return v

    @validator('preselected_tag_id')
    def preselected_tag_id_validate_regular_expression(cls, v):
        if v is None:
            return v

        if not re.match(r"^[a-f0-9]{24}$", v):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return v

    @validator('bit_mask_data')
    def bit_mask_data_validate_regular_expression(cls, v):
        if not re.match(r"^0x[a-f0-9]+$", v):
            raise ValueError(r"must validate the regular expression /^0x[a-f0-9]+$/")
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

    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.dict(by_alias=True))

    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> TagCreateRequest:
        """Create an instance of TagCreateRequest from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of changes
        if self.changes:
            _dict['changes'] = self.changes.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> TagCreateRequest:
        """Create an instance of TagCreateRequest from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return TagCreateRequest.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in TagCreateRequest) in the input: " + str(obj))

        _obj = TagCreateRequest.parse_obj({
            "name": obj.get("name"),
            "prev_tag_id": obj.get("prevTagId"),
            "query_tag_id": obj.get("queryTagId"),
            "preselected_tag_id": obj.get("preselectedTagId"),
            "bit_mask_data": obj.get("bitMaskData"),
            "tot_size": obj.get("totSize"),
            "creator": obj.get("creator"),
            "changes": TagChangeData.from_dict(obj.get("changes")) if obj.get("changes") is not None else None,
            "run_id": obj.get("runId")
        })
        return _obj


