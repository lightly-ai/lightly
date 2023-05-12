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
from pydantic import Extra,  BaseModel, Field, StrictInt, StrictStr

class SamaTaskData(BaseModel):
    """
    SamaTaskData
    """
    id: StrictInt = Field(...)
    url: StrictStr = Field(..., description="A URL which allows anyone in possession of said URL for the time specified by the expiresIn query param to access the resource")
    image: Optional[StrictStr] = Field(None, description="A URL which allows anyone in possession of said URL for the time specified by the expiresIn query param to access the resource")
    lightly_file_name: Optional[StrictStr] = Field(None, alias="lightlyFileName", description="The original fileName of the sample. This is unique within a dataset")
    lightly_meta_info: Optional[StrictStr] = Field(None, alias="lightlyMetaInfo")
    __properties = ["id", "url", "image", "lightlyFileName", "lightlyMetaInfo"]

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
    def from_json(cls, json_str: str) -> SamaTaskData:
        """Create an instance of SamaTaskData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> SamaTaskData:
        """Create an instance of SamaTaskData from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return SamaTaskData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in SamaTaskData) in the input: " + str(obj))

        _obj = SamaTaskData.parse_obj({
            "id": obj.get("id"),
            "url": obj.get("url"),
            "image": obj.get("image"),
            "lightly_file_name": obj.get("lightlyFileName"),
            "lightly_meta_info": obj.get("lightlyMetaInfo")
        })
        return _obj

