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



try:
    # Pydantic >=v1.10.17
    from pydantic.v1 import BaseModel, Field, constr, validator
    pass # Add pass to avoid empty try/except if no imports are generated for this file.
except ImportError:
    # Pydantic v1
    from pydantic import BaseModel, Field, constr, validator
    pass # Add pass to avoid empty try/except if no imports are generated for this file.

class DatasourceConfigOBSAllOf(BaseModel):
    """
    Object Storage Service (OBS) is a S3 (AWS) compatible cloud storage like openstack
    """
    obs_endpoint: constr(strict=True, min_length=1) = Field(..., alias="obsEndpoint", description="The Object Storage Service (OBS) endpoint to use of your S3 compatible cloud storage provider")
    obs_access_key_id: constr(strict=True, min_length=1) = Field(..., alias="obsAccessKeyId", description="The Access Key Id of the credential you are providing Lightly to use")
    obs_secret_access_key: constr(strict=True, min_length=1) = Field(..., alias="obsSecretAccessKey", description="The Secret Access Key of the credential you are providing Lightly to use")
    __properties = ["obsEndpoint", "obsAccessKeyId", "obsSecretAccessKey"]

    @validator('obs_endpoint')
    def obs_endpoint_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^https?:\/\/.+$", value):
            raise ValueError(r"must validate the regular expression /^https?:\/\/.+$/")
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
    def from_json(cls, json_str: str) -> DatasourceConfigOBSAllOf:
        """Create an instance of DatasourceConfigOBSAllOf from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DatasourceConfigOBSAllOf:
        """Create an instance of DatasourceConfigOBSAllOf from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return DatasourceConfigOBSAllOf.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DatasourceConfigOBSAllOf) in the input: " + str(obj))

        _obj = DatasourceConfigOBSAllOf.parse_obj({
            "obs_endpoint": obj.get("obsEndpoint"),
            "obs_access_key_id": obj.get("obsAccessKeyId"),
            "obs_secret_access_key": obj.get("obsSecretAccessKey")
        })
        return _obj

