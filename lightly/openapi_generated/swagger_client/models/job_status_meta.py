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
from pydantic import Extra, BaseModel, Field, StrictBool, StrictInt
from lightly.openapi_generated.swagger_client.models.job_status_upload_method import (
    JobStatusUploadMethod,
)


class JobStatusMeta(BaseModel):
    """
    JobStatusMeta
    """

    total: StrictInt = Field(...)
    processed: StrictInt = Field(...)
    upload_method: Optional[JobStatusUploadMethod] = Field(None, alias="uploadMethod")
    is_registered: Optional[StrictBool] = Field(
        None,
        alias="isRegistered",
        description="Flag which indicates whether the job was registered or not.",
    )
    __properties = ["total", "processed", "uploadMethod", "isRegistered"]

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
    def from_json(cls, json_str: str) -> JobStatusMeta:
        """Create an instance of JobStatusMeta from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias, exclude={}, exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> JobStatusMeta:
        """Create an instance of JobStatusMeta from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return JobStatusMeta.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError(
                    "Error due to additional fields (not defined in JobStatusMeta) in the input: "
                    + str(obj)
                )

        _obj = JobStatusMeta.parse_obj(
            {
                "total": obj.get("total"),
                "processed": obj.get("processed"),
                "upload_method": obj.get("uploadMethod"),
                "is_registered": obj.get("isRegistered"),
            }
        )
        return _obj
