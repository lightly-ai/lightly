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
from pydantic import Extra,  BaseModel, Field, StrictStr, conlist, constr, validator
from lightly.openapi_client.models.creator import Creator
from lightly.openapi_client.models.docker_worker_type import DockerWorkerType

class CreateDockerWorkerRegistryEntryRequest(BaseModel):
    """
    CreateDockerWorkerRegistryEntryRequest
    """
    name: constr(strict=True, min_length=3) = Field(...)
    worker_type: DockerWorkerType = Field(..., alias="workerType")
    labels: Optional[conlist(StrictStr)] = Field(None, description="The labels used for specifying the run-worker-relationship")
    creator: Optional[Creator] = None
    docker_version: Optional[StrictStr] = Field(None, alias="dockerVersion")
    __properties = ["name", "workerType", "labels", "creator", "dockerVersion"]

    @validator('name')
    def name_validate_regular_expression(cls, v):
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9 _-]+$", v):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9][a-zA-Z0-9 _-]+$/")
        return v

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
    def from_json(cls, json_str: str) -> CreateDockerWorkerRegistryEntryRequest:
        """Create an instance of CreateDockerWorkerRegistryEntryRequest from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> CreateDockerWorkerRegistryEntryRequest:
        """Create an instance of CreateDockerWorkerRegistryEntryRequest from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return CreateDockerWorkerRegistryEntryRequest.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in CreateDockerWorkerRegistryEntryRequest) in the input: " + str(obj))

        _obj = CreateDockerWorkerRegistryEntryRequest.parse_obj({
            "name": obj.get("name"),
            "worker_type": obj.get("workerType"),
            "labels": obj.get("labels"),
            "creator": obj.get("creator"),
            "docker_version": obj.get("dockerVersion")
        })
        return _obj


