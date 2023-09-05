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
from pydantic import Extra, BaseModel, Field, StrictStr


class ProfileMeDataSettings(BaseModel):
    """
    ProfileMeDataSettings
    """

    locale: Optional[StrictStr] = Field(
        "en", description="Which locale does the user prefer"
    )
    date_format: Optional[StrictStr] = Field(
        None,
        alias="dateFormat",
        description="Which format for dates does the user prefer",
    )
    number_format: Optional[StrictStr] = Field(
        None,
        alias="numberFormat",
        description="Which format for numbers does the user prefer",
    )
    additional_properties: Dict[str, Any] = {}
    __properties = ["locale", "dateFormat", "numberFormat"]

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
    def from_json(cls, json_str: str) -> ProfileMeDataSettings:
        """Create an instance of ProfileMeDataSettings from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(
            by_alias=by_alias, exclude={"additional_properties"}, exclude_none=True
        )
        # puts key-value pairs in additional_properties in the top level
        if self.additional_properties is not None:
            for _key, _value in self.additional_properties.items():
                _dict[_key] = _value

        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> ProfileMeDataSettings:
        """Create an instance of ProfileMeDataSettings from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return ProfileMeDataSettings.parse_obj(obj)

        _obj = ProfileMeDataSettings.parse_obj(
            {
                "locale": obj.get("locale") if obj.get("locale") is not None else "en",
                "date_format": obj.get("dateFormat"),
                "number_format": obj.get("numberFormat"),
            }
        )
        # store additional fields in additional_properties
        for _key in obj.keys():
            if _key not in cls.__properties:
                _obj.additional_properties[_key] = obj.get(_key)

        return _obj
