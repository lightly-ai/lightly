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


from typing import List
from pydantic import Extra,  BaseModel, Field, StrictStr, conint, conlist, constr, validator
from lightly.openapi_client.models.shared_access_type import SharedAccessType

class SharedAccessConfigData(BaseModel):
    """
    SharedAccessConfigData
    """
    id: constr(strict=True) = Field(..., description="MongoDB ObjectId")
    owner: StrictStr = Field(..., description="Id of the user who owns the dataset")
    access_type: SharedAccessType = Field(..., alias="accessType")
    users: conlist(StrictStr) = Field(..., description="List of user mails with access to the dataset")
    teams: conlist(StrictStr) = Field(..., description="List of teams with access to the dataset")
    created_at: conint(strict=True, ge=0) = Field(..., alias="createdAt", description="unix timestamp in milliseconds")
    last_modified_at: conint(strict=True, ge=0) = Field(..., alias="lastModifiedAt", description="unix timestamp in milliseconds")
    __properties = ["id", "owner", "accessType", "users", "teams", "createdAt", "lastModifiedAt"]

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
    def from_json(cls, json_str: str) -> SharedAccessConfigData:
        """Create an instance of SharedAccessConfigData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> SharedAccessConfigData:
        """Create an instance of SharedAccessConfigData from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return SharedAccessConfigData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in SharedAccessConfigData) in the input: " + str(obj))

        _obj = SharedAccessConfigData.parse_obj({
            "id": obj.get("id"),
            "owner": obj.get("owner"),
            "access_type": obj.get("accessType"),
            "users": obj.get("users"),
            "teams": obj.get("teams"),
            "created_at": obj.get("createdAt"),
            "last_modified_at": obj.get("lastModifiedAt")
        })
        return _obj

