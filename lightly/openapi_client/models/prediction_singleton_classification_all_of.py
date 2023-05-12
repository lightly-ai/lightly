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



from typing import List, Optional, Union
from pydantic import Extra,  BaseModel, Field, confloat, conint, conlist

class PredictionSingletonClassificationAllOf(BaseModel):
    """
    PredictionSingletonClassificationAllOf
    """
    probabilities: Optional[conlist(Union[confloat(le=1, ge=0, strict=True), conint(le=1, ge=0, strict=True)])] = Field(None, description="The probabilities of it being a certain category other than the one which was selected. The sum of all probabilities should equal 1.")
    __properties = ["probabilities"]

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
    def from_json(cls, json_str: str) -> PredictionSingletonClassificationAllOf:
        """Create an instance of PredictionSingletonClassificationAllOf from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> PredictionSingletonClassificationAllOf:
        """Create an instance of PredictionSingletonClassificationAllOf from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return PredictionSingletonClassificationAllOf.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in PredictionSingletonClassificationAllOf) in the input: " + str(obj))

        _obj = PredictionSingletonClassificationAllOf.parse_obj({
            "probabilities": obj.get("probabilities")
        })
        return _obj


