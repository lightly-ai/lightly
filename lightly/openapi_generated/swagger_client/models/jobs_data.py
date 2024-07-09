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
try:
    # Pydantic >=v1.10.17
    from pydantic.v1 import BaseModel, Field, StrictStr, conint, constr, validator
    pass # Add pass to avoid empty try/except if no imports are generated for this file.
except ImportError:
    # Pydantic v1
    from pydantic import BaseModel, Field, StrictStr, conint, constr, validator
    pass # Add pass to avoid empty try/except if no imports are generated for this file.
from lightly.openapi_generated.swagger_client.models.job_result_type import JobResultType
from lightly.openapi_generated.swagger_client.models.job_state import JobState

class JobsData(BaseModel):
    """
    JobsData
    """
    id: constr(strict=True) = Field(..., description="MongoDB ObjectId")
    job_id: StrictStr = Field(..., alias="jobId")
    job_type: JobResultType = Field(..., alias="jobType")
    dataset_id: Optional[constr(strict=True)] = Field(None, alias="datasetId", description="MongoDB ObjectId")
    status: JobState = Field(...)
    finished_at: Optional[conint(strict=True, ge=0)] = Field(None, alias="finishedAt", description="unix timestamp in milliseconds")
    created_at: conint(strict=True, ge=0) = Field(..., alias="createdAt", description="unix timestamp in milliseconds")
    __properties = ["id", "jobId", "jobType", "datasetId", "status", "finishedAt", "createdAt"]

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
    def from_json(cls, json_str: str) -> JobsData:
        """Create an instance of JobsData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> JobsData:
        """Create an instance of JobsData from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return JobsData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in JobsData) in the input: " + str(obj))

        _obj = JobsData.parse_obj({
            "id": obj.get("id"),
            "job_id": obj.get("jobId"),
            "job_type": obj.get("jobType"),
            "dataset_id": obj.get("datasetId"),
            "status": obj.get("status"),
            "finished_at": obj.get("finishedAt"),
            "created_at": obj.get("createdAt")
        })
        return _obj

