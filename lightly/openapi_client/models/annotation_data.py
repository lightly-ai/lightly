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
from pydantic import Extra,  BaseModel, Field, StrictStr, conint
from lightly.openapi_client.models.annotation_meta_data import AnnotationMetaData
from lightly.openapi_client.models.annotation_offer_data import AnnotationOfferData
from lightly.openapi_client.models.annotation_state import AnnotationState

class AnnotationData(BaseModel):
    """
    AnnotationData
    """
    id: StrictStr = Field(..., alias="_id")
    state: AnnotationState = Field(...)
    dataset_id: StrictStr = Field(..., alias="datasetId")
    tag_id: StrictStr = Field(..., alias="tagId")
    partner_id: Optional[StrictStr] = Field(None, alias="partnerId")
    created_at: conint(strict=True, ge=0) = Field(..., alias="createdAt", description="unix timestamp in milliseconds")
    last_modified_at: conint(strict=True, ge=0) = Field(..., alias="lastModifiedAt", description="unix timestamp in milliseconds")
    meta: AnnotationMetaData = Field(...)
    offer: Optional[AnnotationOfferData] = None
    __properties = ["_id", "state", "datasetId", "tagId", "partnerId", "createdAt", "lastModifiedAt", "meta", "offer"]

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
    def from_json(cls, json_str: str) -> AnnotationData:
        """Create an instance of AnnotationData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of meta
        if self.meta:
            _dict['meta' if by_alias else 'meta'] = self.meta.to_dict(by_alias=by_alias)
        # override the default output from pydantic by calling `to_dict()` of offer
        if self.offer:
            _dict['offer' if by_alias else 'offer'] = self.offer.to_dict(by_alias=by_alias)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> AnnotationData:
        """Create an instance of AnnotationData from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return AnnotationData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in AnnotationData) in the input: " + str(obj))

        _obj = AnnotationData.parse_obj({
            "id": obj.get("_id"),
            "state": obj.get("state"),
            "dataset_id": obj.get("datasetId"),
            "tag_id": obj.get("tagId"),
            "partner_id": obj.get("partnerId"),
            "created_at": obj.get("createdAt"),
            "last_modified_at": obj.get("lastModifiedAt"),
            "meta": AnnotationMetaData.from_dict(obj.get("meta")) if obj.get("meta") is not None else None,
            "offer": AnnotationOfferData.from_dict(obj.get("offer")) if obj.get("offer") is not None else None
        })
        return _obj

