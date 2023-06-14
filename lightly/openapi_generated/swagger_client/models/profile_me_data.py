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


from typing import List, Optional, Union
from pydantic import Extra,  BaseModel, Field, StrictFloat, StrictInt, StrictStr, conint, conlist, constr
from lightly.openapi_generated.swagger_client.models.profile_me_data_settings import ProfileMeDataSettings
from lightly.openapi_generated.swagger_client.models.team_basic_data import TeamBasicData
from lightly.openapi_generated.swagger_client.models.user_type import UserType

class ProfileMeData(BaseModel):
    """
    ProfileMeData
    """
    id: StrictStr = Field(...)
    user_type: UserType = Field(..., alias="userType")
    email: StrictStr = Field(..., description="email of the user")
    nickname: Optional[StrictStr] = None
    name: Optional[StrictStr] = None
    given_name: Optional[StrictStr] = Field(None, alias="givenName")
    family_name: Optional[StrictStr] = Field(None, alias="familyName")
    token: Optional[constr(strict=True, min_length=5)] = Field(None, description="The user's token to be used for authentication via token querystring")
    created_at: conint(strict=True, ge=0) = Field(..., alias="createdAt", description="unix timestamp in milliseconds")
    teams: Optional[conlist(TeamBasicData)] = None
    settings: ProfileMeDataSettings = Field(...)
    onboarding: Optional[Union[StrictFloat, StrictInt]] = None
    __properties = ["id", "userType", "email", "nickname", "name", "givenName", "familyName", "token", "createdAt", "teams", "settings", "onboarding"]

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
    def from_json(cls, json_str: str) -> ProfileMeData:
        """Create an instance of ProfileMeData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in teams (list)
        _items = []
        if self.teams:
            for _item in self.teams:
                if _item:
                    _items.append(_item.to_dict(by_alias=by_alias))
            _dict['teams' if by_alias else 'teams'] = _items
        # override the default output from pydantic by calling `to_dict()` of settings
        if self.settings:
            _dict['settings' if by_alias else 'settings'] = self.settings.to_dict(by_alias=by_alias)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> ProfileMeData:
        """Create an instance of ProfileMeData from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return ProfileMeData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in ProfileMeData) in the input: " + str(obj))

        _obj = ProfileMeData.parse_obj({
            "id": obj.get("id"),
            "user_type": obj.get("userType"),
            "email": obj.get("email"),
            "nickname": obj.get("nickname"),
            "name": obj.get("name"),
            "given_name": obj.get("givenName"),
            "family_name": obj.get("familyName"),
            "token": obj.get("token"),
            "created_at": obj.get("createdAt"),
            "teams": [TeamBasicData.from_dict(_item) for _item in obj.get("teams")] if obj.get("teams") is not None else None,
            "settings": ProfileMeDataSettings.from_dict(obj.get("settings")) if obj.get("settings") is not None else None,
            "onboarding": obj.get("onboarding")
        })
        return _obj

