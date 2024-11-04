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


from typing import List, Optional, Union
try:
    # Pydantic >=v1.10.17
    from pydantic.v1 import BaseModel, Field, confloat, conint, conlist, constr
except ImportError:
    # Pydantic v1
    from pydantic import BaseModel, Field, confloat, conint, conlist, constr
from lightly.openapi_generated.swagger_client.models.auto_task import AutoTask
from lightly.openapi_generated.swagger_client.models.selection_config_v4_entry import SelectionConfigV4Entry

class SelectionConfigV4(BaseModel):
    """
    SelectionConfigV4
    """
    n_samples: Optional[conint(strict=True, ge=-1)] = Field(None, alias="nSamples")
    proportion_samples: Optional[Union[confloat(le=1.0, ge=0.0, strict=True), conint(le=1, ge=0, strict=True)]] = Field(None, alias="proportionSamples")
    strategies: conlist(SelectionConfigV4Entry, min_items=1) = Field(...)
    lightly_path_regex: Optional[constr(strict=True, min_length=1)] = Field(None, alias="lightlyPathRegex", description="The Lightly Path Regex to extract information from filenames for metadata balancing and more. Docs are coming soon.")
    auto_tasks: Optional[conlist(AutoTask)] = Field(None, alias="autoTasks", description="Array of tasks we automatically apply to images create fake prediction for. These can then be used with the selection strategies.")
    __properties = ["nSamples", "proportionSamples", "strategies", "lightlyPathRegex", "autoTasks"]

    class Config:
        """Pydantic configuration"""
        allow_population_by_field_name = True
        validate_assignment = True
        use_enum_values = True
        extra = "forbid"

    def to_str(self, by_alias: bool = False) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.dict(by_alias=by_alias))

    def to_json(self, by_alias: bool = False) -> str:
        """Returns the JSON representation of the model"""
        return json.dumps(self.to_dict(by_alias=by_alias))

    @classmethod
    def from_json(cls, json_str: str) -> SelectionConfigV4:
        """Create an instance of SelectionConfigV4 from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in strategies (list)
        _items = []
        if self.strategies:
            for _item in self.strategies:
                if _item:
                    _items.append(_item.to_dict(by_alias=by_alias))
            _dict['strategies' if by_alias else 'strategies'] = _items
        # override the default output from pydantic by calling `to_dict()` of each item in auto_tasks (list)
        _items = []
        if self.auto_tasks:
            for _item in self.auto_tasks:
                if _item:
                    _items.append(_item.to_dict(by_alias=by_alias))
            _dict['autoTasks' if by_alias else 'auto_tasks'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> SelectionConfigV4:
        """Create an instance of SelectionConfigV4 from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return SelectionConfigV4.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in SelectionConfigV4) in the input: " + str(obj))

        _obj = SelectionConfigV4.parse_obj({
            "n_samples": obj.get("nSamples"),
            "proportion_samples": obj.get("proportionSamples"),
            "strategies": [SelectionConfigV4Entry.from_dict(_item) for _item in obj.get("strategies")] if obj.get("strategies") is not None else None,
            "lightly_path_regex": obj.get("lightlyPathRegex"),
            "auto_tasks": [AutoTask.from_dict(_item) for _item in obj.get("autoTasks")] if obj.get("autoTasks") is not None else None
        })
        return _obj

