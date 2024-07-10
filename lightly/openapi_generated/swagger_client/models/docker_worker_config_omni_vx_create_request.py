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
import json
import pprint
import re  # noqa: F401

from typing import Any, List, Optional
try:
    # Pydantic >=v1.10.17
    from pydantic.v1 import BaseModel, Field, StrictStr, ValidationError, validator
except ImportError:
    # Pydantic v1
    from pydantic import BaseModel, Field, StrictStr, ValidationError, validator
from lightly.openapi_generated.swagger_client.models.docker_worker_config_omni_v2_create_request import DockerWorkerConfigOmniV2CreateRequest
from lightly.openapi_generated.swagger_client.models.docker_worker_config_omni_v3_create_request import DockerWorkerConfigOmniV3CreateRequest
from lightly.openapi_generated.swagger_client.models.docker_worker_config_omni_v4_create_request import DockerWorkerConfigOmniV4CreateRequest
try:
    # Pydantic >=v1.10.17
    from pydantic.v1 import StrictStr, Field
except ImportError:
    # Pydantic v1
    from pydantic import StrictStr, Field

DOCKERWORKERCONFIGOMNIVXCREATEREQUEST_ONE_OF_SCHEMAS = ["DockerWorkerConfigOmniV2CreateRequest", "DockerWorkerConfigOmniV3CreateRequest", "DockerWorkerConfigOmniV4CreateRequest"]

class DockerWorkerConfigOmniVXCreateRequest(BaseModel):
    """
    DockerWorkerConfigOmniVXCreateRequest
    """
    # data type: DockerWorkerConfigOmniV2CreateRequest
    oneof_schema_1_validator: Optional[DockerWorkerConfigOmniV2CreateRequest] = None
    # data type: DockerWorkerConfigOmniV3CreateRequest
    oneof_schema_2_validator: Optional[DockerWorkerConfigOmniV3CreateRequest] = None
    # data type: DockerWorkerConfigOmniV4CreateRequest
    oneof_schema_3_validator: Optional[DockerWorkerConfigOmniV4CreateRequest] = None
    actual_instance: Any
    one_of_schemas: List[str] = Field(DOCKERWORKERCONFIGOMNIVXCREATEREQUEST_ONE_OF_SCHEMAS, const=True)

    class Config:
        validate_assignment = True
        use_enum_values = True
        extra = "forbid"

    discriminator_value_class_map = {
    }

    def __init__(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise ValueError("If a position argument is used, only 1 is allowed to set `actual_instance`")
            if kwargs:
                raise ValueError("If a position argument is used, keyword arguments cannot be used.")
            super().__init__(actual_instance=args[0])
        else:
            super().__init__(**kwargs)

    @validator('actual_instance')
    def actual_instance_must_validate_oneof(cls, v):
        instance = DockerWorkerConfigOmniVXCreateRequest.construct()
        error_messages = []
        match = 0
        # validate data type: DockerWorkerConfigOmniV2CreateRequest
        if not isinstance(v, DockerWorkerConfigOmniV2CreateRequest):
            error_messages.append(f"Error! Input type `{type(v)}` is not `DockerWorkerConfigOmniV2CreateRequest`")
        else:
            match += 1
        # validate data type: DockerWorkerConfigOmniV3CreateRequest
        if not isinstance(v, DockerWorkerConfigOmniV3CreateRequest):
            error_messages.append(f"Error! Input type `{type(v)}` is not `DockerWorkerConfigOmniV3CreateRequest`")
        else:
            match += 1
        # validate data type: DockerWorkerConfigOmniV4CreateRequest
        if not isinstance(v, DockerWorkerConfigOmniV4CreateRequest):
            error_messages.append(f"Error! Input type `{type(v)}` is not `DockerWorkerConfigOmniV4CreateRequest`")
        else:
            match += 1
        if match > 1:
            # more than 1 match
            raise ValueError("Multiple matches found when setting `actual_instance` in DockerWorkerConfigOmniVXCreateRequest with oneOf schemas: DockerWorkerConfigOmniV2CreateRequest, DockerWorkerConfigOmniV3CreateRequest, DockerWorkerConfigOmniV4CreateRequest. Details: " + ", ".join(error_messages))
        elif match == 0:
            # no match
            raise ValueError("No match found when setting `actual_instance` in DockerWorkerConfigOmniVXCreateRequest with oneOf schemas: DockerWorkerConfigOmniV2CreateRequest, DockerWorkerConfigOmniV3CreateRequest, DockerWorkerConfigOmniV4CreateRequest. Details: " + ", ".join(error_messages))
        else:
            return v

    @classmethod
    def from_dict(cls, obj: dict) -> DockerWorkerConfigOmniVXCreateRequest:
        return cls.from_json(json.dumps(obj))

    @classmethod
    def from_json(cls, json_str: str) -> DockerWorkerConfigOmniVXCreateRequest:
        """Returns the object represented by the json string"""
        instance = DockerWorkerConfigOmniVXCreateRequest.construct()
        error_messages = []
        match = 0

        # use oneOf discriminator to lookup the data type
        _data_type = json.loads(json_str).get("version")
        if not _data_type:
            raise ValueError("Failed to lookup data type from the field `version` in the input.")

        # check if data type is `DockerWorkerConfigOmniV2CreateRequest`
        if _data_type == "DockerWorkerConfigOmniV2CreateRequest":
            instance.actual_instance = DockerWorkerConfigOmniV2CreateRequest.from_json(json_str)
            return instance

        # check if data type is `DockerWorkerConfigOmniV3CreateRequest`
        if _data_type == "DockerWorkerConfigOmniV3CreateRequest":
            instance.actual_instance = DockerWorkerConfigOmniV3CreateRequest.from_json(json_str)
            return instance

        # check if data type is `DockerWorkerConfigOmniV4CreateRequest`
        if _data_type == "DockerWorkerConfigOmniV4CreateRequest":
            instance.actual_instance = DockerWorkerConfigOmniV4CreateRequest.from_json(json_str)
            return instance

        # check if data type is `DockerWorkerConfigOmniV2CreateRequest`
        if _data_type == "V2":
            instance.actual_instance = DockerWorkerConfigOmniV2CreateRequest.from_json(json_str)
            return instance

        # check if data type is `DockerWorkerConfigOmniV3CreateRequest`
        if _data_type == "V3":
            instance.actual_instance = DockerWorkerConfigOmniV3CreateRequest.from_json(json_str)
            return instance

        # check if data type is `DockerWorkerConfigOmniV4CreateRequest`
        if _data_type == "V4":
            instance.actual_instance = DockerWorkerConfigOmniV4CreateRequest.from_json(json_str)
            return instance

        # deserialize data into DockerWorkerConfigOmniV2CreateRequest
        try:
            instance.actual_instance = DockerWorkerConfigOmniV2CreateRequest.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into DockerWorkerConfigOmniV3CreateRequest
        try:
            instance.actual_instance = DockerWorkerConfigOmniV3CreateRequest.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into DockerWorkerConfigOmniV4CreateRequest
        try:
            instance.actual_instance = DockerWorkerConfigOmniV4CreateRequest.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))

        if match > 1:
            # more than 1 match
            raise ValueError("Multiple matches found when deserializing the JSON string into DockerWorkerConfigOmniVXCreateRequest with oneOf schemas: DockerWorkerConfigOmniV2CreateRequest, DockerWorkerConfigOmniV3CreateRequest, DockerWorkerConfigOmniV4CreateRequest. Details: " + ", ".join(error_messages))
        elif match == 0:
            # no match
            raise ValueError("No match found when deserializing the JSON string into DockerWorkerConfigOmniVXCreateRequest with oneOf schemas: DockerWorkerConfigOmniV2CreateRequest, DockerWorkerConfigOmniV3CreateRequest, DockerWorkerConfigOmniV4CreateRequest. Details: " + ", ".join(error_messages))
        else:
            return instance

    def to_json(self, by_alias: bool = False) -> str:
        """Returns the JSON representation of the actual instance"""
        if self.actual_instance is None:
            return "null"

        to_json = getattr(self.actual_instance, "to_json", None)
        if callable(to_json):
            return self.actual_instance.to_json(by_alias=by_alias)
        else:
            return json.dumps(self.actual_instance)

    def to_dict(self, by_alias: bool = False) -> dict:
        """Returns the dict representation of the actual instance"""
        if self.actual_instance is None:
            return None

        to_dict = getattr(self.actual_instance, "to_dict", None)
        if callable(to_dict):
            return self.actual_instance.to_dict(by_alias=by_alias)
        else:
            # primitive type
            return self.actual_instance

    def to_str(self, by_alias: bool = False) -> str:
        """Returns the string representation of the actual instance"""
        return pprint.pformat(self.dict(by_alias=by_alias))

