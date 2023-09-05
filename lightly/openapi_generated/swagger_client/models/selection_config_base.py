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


from typing import Optional, Union
from pydantic import Extra, BaseModel, Field, confloat, conint


class SelectionConfigBase(BaseModel):
    """
    SelectionConfigBase
    """

    n_samples: Optional[conint(strict=True, ge=-1)] = Field(None, alias="nSamples")
    proportion_samples: Optional[
        Union[confloat(le=1.0, ge=0.0, strict=True), conint(le=1, ge=0, strict=True)]
    ] = Field(None, alias="proportionSamples")
    __properties = ["nSamples", "proportionSamples"]

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
    def from_json(cls, json_str: str) -> SelectionConfigBase:
        """Create an instance of SelectionConfigBase from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias, exclude={}, exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> SelectionConfigBase:
        """Create an instance of SelectionConfigBase from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return SelectionConfigBase.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError(
                    "Error due to additional fields (not defined in SelectionConfigBase) in the input: "
                    + str(obj)
                )

        _obj = SelectionConfigBase.parse_obj(
            {
                "n_samples": obj.get("nSamples"),
                "proportion_samples": obj.get("proportionSamples"),
            }
        )
        return _obj
