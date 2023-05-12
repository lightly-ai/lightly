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



from typing import Optional, Union
from pydantic import Extra,  BaseModel, Field, StrictFloat, StrictInt, constr, validator

class TagChangeDataUpsize(BaseModel):
    """
    TagChangeDataUpsize
    """
    run_id: Optional[constr(strict=True)] = Field(None, alias="runId", description="MongoDB ObjectId")
    var_from: Union[StrictFloat, StrictInt] = Field(..., alias="from")
    to: Union[StrictFloat, StrictInt] = Field(...)
    __properties = ["runId", "from", "to"]

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
    def from_json(cls, json_str: str) -> TagChangeDataUpsize:
        """Create an instance of TagChangeDataUpsize from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> TagChangeDataUpsize:
        """Create an instance of TagChangeDataUpsize from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return TagChangeDataUpsize.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in TagChangeDataUpsize) in the input: " + str(obj))

        _obj = TagChangeDataUpsize.parse_obj({
            "run_id": obj.get("runId"),
            "var_from": obj.get("from"),
            "to": obj.get("to")
        })
        return _obj


