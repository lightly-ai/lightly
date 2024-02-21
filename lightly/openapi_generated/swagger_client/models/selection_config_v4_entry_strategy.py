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


from typing import Any, Dict, Optional, Union
from pydantic import Extra,  BaseModel, Field, StrictFloat, StrictInt, confloat, conint
from lightly.openapi_generated.swagger_client.models.selection_config_v3_entry_strategy_all_of_target_range import SelectionConfigV3EntryStrategyAllOfTargetRange
from lightly.openapi_generated.swagger_client.models.selection_strategy_threshold_operation import SelectionStrategyThresholdOperation
from lightly.openapi_generated.swagger_client.models.selection_strategy_type_v3 import SelectionStrategyTypeV3

class SelectionConfigV4EntryStrategy(BaseModel):
    """
    SelectionConfigV4EntryStrategy
    """
    type: SelectionStrategyTypeV3 = Field(...)
    stopping_condition_minimum_distance: Optional[Union[StrictFloat, StrictInt]] = None
    threshold: Optional[Union[StrictFloat, StrictInt]] = None
    operation: Optional[SelectionStrategyThresholdOperation] = None
    target: Optional[Dict[str, Any]] = None
    num_nearest_neighbors: Optional[Union[confloat(ge=2, strict=True), conint(ge=2, strict=True)]] = Field(None, alias="numNearestNeighbors", description="It is the number of nearest datapoints used to compute the typicality of each sample. ")
    stopping_condition_minimum_typicality: Optional[Union[confloat(gt=0, strict=True), conint(gt=0, strict=True)]] = Field(None, alias="stoppingConditionMinimumTypicality", description="It is the minimal allowed typicality of the selected samples. When the typicality of the selected samples reaches this, the selection stops. It should be  a number between 0 and 1. ")
    strength: Optional[Union[confloat(le=1000000000, ge=-1000000000, strict=True), conint(le=1000000000, ge=-1000000000, strict=True)]] = Field(None, description="The relative strength of this strategy compared to other strategies. The default value is 1.0, which is set in the worker for backwards compatibility. The minimum and maximum values of +-10^9 are used to prevent numerical issues. ")
    stopping_condition_max_sum: Optional[Union[confloat(ge=0.0, strict=True), conint(ge=0, strict=True)]] = Field(None, alias="stoppingConditionMaxSum", description="When the sum of inputs reaches this, the selection stops. Only compatible with the WEIGHTS strategy. Similar to the stopping_condition_minimum_distance for the DIVERSITY strategy. ")
    target_range: Optional[SelectionConfigV3EntryStrategyAllOfTargetRange] = Field(None, alias="targetRange")
    __properties = ["type", "stopping_condition_minimum_distance", "threshold", "operation", "target", "numNearestNeighbors", "stoppingConditionMinimumTypicality", "strength", "stoppingConditionMaxSum", "targetRange"]

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
    def from_json(cls, json_str: str) -> SelectionConfigV4EntryStrategy:
        """Create an instance of SelectionConfigV4EntryStrategy from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of target_range
        if self.target_range:
            _dict['targetRange' if by_alias else 'target_range'] = self.target_range.to_dict(by_alias=by_alias)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> SelectionConfigV4EntryStrategy:
        """Create an instance of SelectionConfigV4EntryStrategy from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return SelectionConfigV4EntryStrategy.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in SelectionConfigV4EntryStrategy) in the input: " + str(obj))

        _obj = SelectionConfigV4EntryStrategy.parse_obj({
            "type": obj.get("type"),
            "stopping_condition_minimum_distance": obj.get("stopping_condition_minimum_distance"),
            "threshold": obj.get("threshold"),
            "operation": obj.get("operation"),
            "target": obj.get("target"),
            "num_nearest_neighbors": obj.get("numNearestNeighbors"),
            "stopping_condition_minimum_typicality": obj.get("stoppingConditionMinimumTypicality"),
            "strength": obj.get("strength"),
            "stopping_condition_max_sum": obj.get("stoppingConditionMaxSum"),
            "target_range": SelectionConfigV3EntryStrategyAllOfTargetRange.from_dict(obj.get("targetRange")) if obj.get("targetRange") is not None else None
        })
        return _obj

