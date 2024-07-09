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


from typing import List, Optional
try:
    # Pydantic >=v1.10.17
    from pydantic.v1 import BaseModel, Field, StrictBool, StrictStr, conint, conlist, constr, validator
    pass # Add pass to avoid empty try/except if no imports are generated for this file.
except ImportError:
    # Pydantic v1
    from pydantic import BaseModel, Field, StrictBool, StrictStr, conint, conlist, constr, validator
    pass # Add pass to avoid empty try/except if no imports are generated for this file.
from lightly.openapi_generated.swagger_client.models.docker_run_artifact_data import DockerRunArtifactData
from lightly.openapi_generated.swagger_client.models.docker_run_state import DockerRunState

class DockerRunData(BaseModel):
    """
    DockerRunData
    """
    id: constr(strict=True) = Field(..., description="MongoDB ObjectId")
    user_id: StrictStr = Field(..., alias="userId")
    docker_version: StrictStr = Field(..., alias="dockerVersion")
    state: DockerRunState = Field(...)
    archived: Optional[StrictBool] = Field(None, description="if the run is archived")
    dataset_id: Optional[constr(strict=True)] = Field(None, alias="datasetId", description="MongoDB ObjectId")
    config_id: Optional[constr(strict=True)] = Field(None, alias="configId", description="MongoDB ObjectId")
    scheduled_id: Optional[constr(strict=True)] = Field(None, alias="scheduledId", description="MongoDB ObjectId")
    created_at: conint(strict=True, ge=0) = Field(..., alias="createdAt", description="unix timestamp in milliseconds")
    last_modified_at: conint(strict=True, ge=0) = Field(..., alias="lastModifiedAt", description="unix timestamp in milliseconds")
    message: Optional[StrictStr] = Field(None, description="last message sent to the docker run")
    artifacts: Optional[conlist(DockerRunArtifactData)] = Field(None, description="list of artifacts that were created for a run")
    __properties = ["id", "userId", "dockerVersion", "state", "archived", "datasetId", "configId", "scheduledId", "createdAt", "lastModifiedAt", "message", "artifacts"]

    @validator('id')
    def id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator('dataset_id')
    def dataset_id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator('config_id')
    def config_id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator('scheduled_id')
    def scheduled_id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

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
    def from_json(cls, json_str: str) -> DockerRunData:
        """Create an instance of DockerRunData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in artifacts (list)
        _items = []
        if self.artifacts:
            for _item in self.artifacts:
                if _item:
                    _items.append(_item.to_dict(by_alias=by_alias))
            _dict['artifacts' if by_alias else 'artifacts'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DockerRunData:
        """Create an instance of DockerRunData from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return DockerRunData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DockerRunData) in the input: " + str(obj))

        _obj = DockerRunData.parse_obj({
            "id": obj.get("id"),
            "user_id": obj.get("userId"),
            "docker_version": obj.get("dockerVersion"),
            "state": obj.get("state"),
            "archived": obj.get("archived"),
            "dataset_id": obj.get("datasetId"),
            "config_id": obj.get("configId"),
            "scheduled_id": obj.get("scheduledId"),
            "created_at": obj.get("createdAt"),
            "last_modified_at": obj.get("lastModifiedAt"),
            "message": obj.get("message"),
            "artifacts": [DockerRunArtifactData.from_dict(_item) for _item in obj.get("artifacts")] if obj.get("artifacts") is not None else None
        })
        return _obj

