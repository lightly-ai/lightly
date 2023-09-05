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


from pydantic import Extra, BaseModel, Field, StrictStr, constr
from lightly.openapi_generated.swagger_client.models.datasource_config_base import (
    DatasourceConfigBase,
)


class DatasourceConfigAzure(DatasourceConfigBase):
    """
    DatasourceConfigAzure
    """

    full_path: StrictStr = Field(
        ...,
        alias="fullPath",
        description="path includes the bucket name and the path within the bucket where you have stored your information",
    )
    account_name: constr(strict=True, min_length=1) = Field(
        ..., alias="accountName", description="name of the Azure Storage Account"
    )
    account_key: constr(strict=True, min_length=1) = Field(
        ..., alias="accountKey", description="key of the Azure Storage Account"
    )
    __properties = [
        "id",
        "purpose",
        "type",
        "thumbSuffix",
        "fullPath",
        "accountName",
        "accountKey",
    ]

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
    def from_json(cls, json_str: str) -> DatasourceConfigAzure:
        """Create an instance of DatasourceConfigAzure from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias, exclude={}, exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DatasourceConfigAzure:
        """Create an instance of DatasourceConfigAzure from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return DatasourceConfigAzure.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError(
                    "Error due to additional fields (not defined in DatasourceConfigAzure) in the input: "
                    + str(obj)
                )

        _obj = DatasourceConfigAzure.parse_obj(
            {
                "id": obj.get("id"),
                "purpose": obj.get("purpose"),
                "type": obj.get("type"),
                "thumb_suffix": obj.get("thumbSuffix"),
                "full_path": obj.get("fullPath"),
                "account_name": obj.get("accountName"),
                "account_key": obj.get("accountKey"),
            }
        )
        return _obj
