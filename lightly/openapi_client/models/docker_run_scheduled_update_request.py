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



from typing import List, Optional
from pydantic import Extra,  BaseModel, Field, StrictStr, conlist
from lightly.openapi_client.models.docker_run_scheduled_priority import DockerRunScheduledPriority
from lightly.openapi_client.models.docker_run_scheduled_state import DockerRunScheduledState

class DockerRunScheduledUpdateRequest(BaseModel):
    """
    DockerRunScheduledUpdateRequest
    """
    state: DockerRunScheduledState = Field(...)
    priority: Optional[DockerRunScheduledPriority] = None
    runs_on: Optional[conlist(StrictStr)] = Field(None, alias="runsOn", description="The labels used for specifying the run-worker-relationship")
    __properties = ["state", "priority", "runsOn"]

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
    def from_json(cls, json_str: str) -> DockerRunScheduledUpdateRequest:
        """Create an instance of DockerRunScheduledUpdateRequest from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DockerRunScheduledUpdateRequest:
        """Create an instance of DockerRunScheduledUpdateRequest from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return DockerRunScheduledUpdateRequest.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DockerRunScheduledUpdateRequest) in the input: " + str(obj))

        _obj = DockerRunScheduledUpdateRequest.parse_obj({
            "state": obj.get("state"),
            "priority": obj.get("priority"),
            "runs_on": obj.get("runsOn")
        })
        return _obj

