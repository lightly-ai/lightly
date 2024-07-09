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
    from pydantic.v1 import BaseModel, Field, constr, validator
    pass # Add pass to avoid empty try/except if no imports are generated for this file.
except ImportError:
    # Pydantic v1
    from pydantic import BaseModel, Field, constr, validator
    pass # Add pass to avoid empty try/except if no imports are generated for this file.
from lightly.openapi_generated.swagger_client.models.s3_region import S3Region

class DatasourceConfigS3DelegatedAccessAllOf(BaseModel):
    """
    DatasourceConfigS3DelegatedAccessAllOf
    """
    s3_region: S3Region = Field(..., alias="s3Region")
    s3_external_id: constr(strict=True, min_length=10) = Field(..., alias="s3ExternalId", description="The external ID specified when creating the role. More information can be found here: - https://docs.aws.amazon.com/IAM/latest/UserGuide/confused-deputy.html - https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_iam-condition-keys.html#ck_externalid ")
    s3_arn: constr(strict=True, min_length=12) = Field(..., alias="s3ARN", description="The ARN of the role you created")
    s3_server_side_encryption_kms_key: Optional[constr(strict=True, min_length=1)] = Field(None, alias="s3ServerSideEncryptionKMSKey", description="If set, Lightly Worker will automatically set the headers to use server side encryption https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingKMSEncryption.html with this value as the appropriate KMS key arn. This will encrypt the files created by Lightly (crops, frames, thumbnails) in the S3 bucket. ")
    __properties = ["s3Region", "s3ExternalId", "s3ARN", "s3ServerSideEncryptionKMSKey"]

    @validator('s3_external_id')
    def s3_external_id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-zA-Z0-9_+=,.@:\/-]+$", value):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9_+=,.@:\/-]+$/")
        return value

    @validator('s3_arn')
    def s3_arn_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^arn:aws:iam::[0-9]{12}:role.+$", value):
            raise ValueError(r"must validate the regular expression /^arn:aws:iam::[0-9]{12}:role.+$/")
        return value

    @validator('s3_server_side_encryption_kms_key')
    def s3_server_side_encryption_kms_key_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^arn:aws:kms:[a-zA-Z0-9-]*:[0-9]{12}:key.+$", value):
            raise ValueError(r"must validate the regular expression /^arn:aws:kms:[a-zA-Z0-9-]*:[0-9]{12}:key.+$/")
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
    def from_json(cls, json_str: str) -> DatasourceConfigS3DelegatedAccessAllOf:
        """Create an instance of DatasourceConfigS3DelegatedAccessAllOf from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DatasourceConfigS3DelegatedAccessAllOf:
        """Create an instance of DatasourceConfigS3DelegatedAccessAllOf from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return DatasourceConfigS3DelegatedAccessAllOf.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DatasourceConfigS3DelegatedAccessAllOf) in the input: " + str(obj))

        _obj = DatasourceConfigS3DelegatedAccessAllOf.parse_obj({
            "s3_region": obj.get("s3Region"),
            "s3_external_id": obj.get("s3ExternalId"),
            "s3_arn": obj.get("s3ARN"),
            "s3_server_side_encryption_kms_key": obj.get("s3ServerSideEncryptionKMSKey")
        })
        return _obj

