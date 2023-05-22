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



from pydantic import Extra,  BaseModel, Field
from lightly.openapi_client.models.selection_config_entry_input import SelectionConfigEntryInput
from lightly.openapi_client.models.selection_config_entry_strategy import SelectionConfigEntryStrategy

class SelectionConfigEntry(BaseModel):
    """
    SelectionConfigEntry
    """
    input: SelectionConfigEntryInput = Field(...)
    strategy: SelectionConfigEntryStrategy = Field(...)
    __properties = ["input", "strategy"]

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
    def from_json(cls, json_str: str) -> SelectionConfigEntry:
        """Create an instance of SelectionConfigEntry from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of input
        if self.input:
            _dict['input' if by_alias else 'input'] = self.input.to_dict(by_alias=by_alias)
        # override the default output from pydantic by calling `to_dict()` of strategy
        if self.strategy:
            _dict['strategy' if by_alias else 'strategy'] = self.strategy.to_dict(by_alias=by_alias)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> SelectionConfigEntry:
        """Create an instance of SelectionConfigEntry from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return SelectionConfigEntry.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in SelectionConfigEntry) in the input: " + str(obj))

        _obj = SelectionConfigEntry.parse_obj({
            "input": SelectionConfigEntryInput.from_dict(obj.get("input")) if obj.get("input") is not None else None,
            "strategy": SelectionConfigEntryStrategy.from_dict(obj.get("strategy")) if obj.get("strategy") is not None else None
        })
        return _obj

