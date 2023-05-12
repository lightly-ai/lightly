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
from pydantic import Extra,  BaseModel, Field, conint
from lightly.openapi_client.models.lightly_model_v3 import LightlyModelV3

class DockerWorkerConfigV3LightlyModel(BaseModel):
    """
    DockerWorkerConfigV3LightlyModel
    """
    name: Optional[LightlyModelV3] = None
    out_dim: Optional[conint(strict=True, ge=1)] = Field(None, alias="outDim")
    num_ftrs: Optional[conint(strict=True, ge=1)] = Field(None, alias="numFtrs")
    width: Optional[conint(strict=True, ge=1)] = None
    __properties = ["name", "outDim", "numFtrs", "width"]

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
    def from_json(cls, json_str: str) -> DockerWorkerConfigV3LightlyModel:
        """Create an instance of DockerWorkerConfigV3LightlyModel from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DockerWorkerConfigV3LightlyModel:
        """Create an instance of DockerWorkerConfigV3LightlyModel from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return DockerWorkerConfigV3LightlyModel.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DockerWorkerConfigV3LightlyModel) in the input: " + str(obj))

        _obj = DockerWorkerConfigV3LightlyModel.parse_obj({
            "name": obj.get("name"),
            "out_dim": obj.get("outDim"),
            "num_ftrs": obj.get("numFtrs"),
            "width": obj.get("width")
        })
        return _obj


