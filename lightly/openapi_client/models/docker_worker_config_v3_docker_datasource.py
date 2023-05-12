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
from pydantic import Extra,  BaseModel, Field, StrictBool

class DockerWorkerConfigV3DockerDatasource(BaseModel):
    """
    DockerWorkerConfigV3DockerDatasource
    """
    bypass_verify: Optional[StrictBool] = Field(None, alias="bypassVerify")
    enable_datapool_update: Optional[StrictBool] = Field(None, alias="enableDatapoolUpdate")
    process_all: Optional[StrictBool] = Field(None, alias="processAll")
    __properties = ["bypassVerify", "enableDatapoolUpdate", "processAll"]

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
    def from_json(cls, json_str: str) -> DockerWorkerConfigV3DockerDatasource:
        """Create an instance of DockerWorkerConfigV3DockerDatasource from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DockerWorkerConfigV3DockerDatasource:
        """Create an instance of DockerWorkerConfigV3DockerDatasource from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return DockerWorkerConfigV3DockerDatasource.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DockerWorkerConfigV3DockerDatasource) in the input: " + str(obj))

        _obj = DockerWorkerConfigV3DockerDatasource.parse_obj({
            "bypass_verify": obj.get("bypassVerify"),
            "enable_datapool_update": obj.get("enableDatapoolUpdate"),
            "process_all": obj.get("processAll")
        })
        return _obj


