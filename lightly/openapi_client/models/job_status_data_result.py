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



from typing import Any, Optional
from pydantic import Extra,  BaseModel, Field
from lightly.openapi_client.models.job_result_type import JobResultType

class JobStatusDataResult(BaseModel):
    """
    JobStatusDataResult
    """
    type: JobResultType = Field(...)
    data: Optional[Any] = Field(None, description="Depending on the job type, this can be anything")
    __properties = ["type", "data"]

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
    def from_json(cls, json_str: str) -> JobStatusDataResult:
        """Create an instance of JobStatusDataResult from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # set to None if data (nullable) is None
        # and __fields_set__ contains the field
        if self.data is None and "data" in self.__fields_set__:
            _dict['data'] = None

        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> JobStatusDataResult:
        """Create an instance of JobStatusDataResult from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return JobStatusDataResult.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in JobStatusDataResult) in the input: " + str(obj))

        _obj = JobStatusDataResult.parse_obj({
            "type": obj.get("type"),
            "data": obj.get("data")
        })
        return _obj


