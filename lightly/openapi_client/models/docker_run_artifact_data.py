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
from pydantic import Extra,  BaseModel, Field, StrictStr, conint, constr, validator
from lightly.openapi_client.models.docker_run_artifact_storage_location import DockerRunArtifactStorageLocation
from lightly.openapi_client.models.docker_run_artifact_type import DockerRunArtifactType

class DockerRunArtifactData(BaseModel):
    """
    DockerRunArtifactData
    """
    id: constr(strict=True) = Field(..., description="MongoDB ObjectId")
    type: DockerRunArtifactType = Field(...)
    file_name: StrictStr = Field(..., alias="fileName")
    storage_location: Optional[DockerRunArtifactStorageLocation] = Field(None, alias="storageLocation")
    created_at: Optional[conint(strict=True, ge=0)] = Field(None, alias="createdAt", description="unix timestamp in milliseconds")
    __properties = ["id", "type", "fileName", "storageLocation", "createdAt"]

    @validator('id')
    def id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

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
    def from_json(cls, json_str: str) -> DockerRunArtifactData:
        """Create an instance of DockerRunArtifactData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DockerRunArtifactData:
        """Create an instance of DockerRunArtifactData from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return DockerRunArtifactData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DockerRunArtifactData) in the input: " + str(obj))

        _obj = DockerRunArtifactData.parse_obj({
            "id": obj.get("id"),
            "type": obj.get("type"),
            "file_name": obj.get("fileName"),
            "storage_location": obj.get("storageLocation"),
            "created_at": obj.get("createdAt")
        })
        return _obj

