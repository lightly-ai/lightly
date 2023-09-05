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
from pydantic import Extra, BaseModel, Field, StrictFloat, StrictInt
from lightly.openapi_generated.swagger_client.models.internal_debug_latency_mongodb import (
    InternalDebugLatencyMongodb,
)


class InternalDebugLatency(BaseModel):
    """
    InternalDebugLatency
    """

    express: Optional[Union[StrictFloat, StrictInt]] = None
    mongodb: Optional[InternalDebugLatencyMongodb] = None
    redis_cache: Optional[InternalDebugLatencyMongodb] = Field(None, alias="redisCache")
    redis_worker: Optional[InternalDebugLatencyMongodb] = Field(
        None, alias="redisWorker"
    )
    __properties = ["express", "mongodb", "redisCache", "redisWorker"]

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
    def from_json(cls, json_str: str) -> InternalDebugLatency:
        """Create an instance of InternalDebugLatency from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias, exclude={}, exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of mongodb
        if self.mongodb:
            _dict["mongodb" if by_alias else "mongodb"] = self.mongodb.to_dict(
                by_alias=by_alias
            )
        # override the default output from pydantic by calling `to_dict()` of redis_cache
        if self.redis_cache:
            _dict[
                "redisCache" if by_alias else "redis_cache"
            ] = self.redis_cache.to_dict(by_alias=by_alias)
        # override the default output from pydantic by calling `to_dict()` of redis_worker
        if self.redis_worker:
            _dict[
                "redisWorker" if by_alias else "redis_worker"
            ] = self.redis_worker.to_dict(by_alias=by_alias)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> InternalDebugLatency:
        """Create an instance of InternalDebugLatency from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return InternalDebugLatency.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError(
                    "Error due to additional fields (not defined in InternalDebugLatency) in the input: "
                    + str(obj)
                )

        _obj = InternalDebugLatency.parse_obj(
            {
                "express": obj.get("express"),
                "mongodb": InternalDebugLatencyMongodb.from_dict(obj.get("mongodb"))
                if obj.get("mongodb") is not None
                else None,
                "redis_cache": InternalDebugLatencyMongodb.from_dict(
                    obj.get("redisCache")
                )
                if obj.get("redisCache") is not None
                else None,
                "redis_worker": InternalDebugLatencyMongodb.from_dict(
                    obj.get("redisWorker")
                )
                if obj.get("redisWorker") is not None
                else None,
            }
        )
        return _obj
