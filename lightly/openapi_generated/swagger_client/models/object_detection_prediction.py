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
try:
    # Pydantic >=v1.10.17
    from pydantic.v1 import BaseModel, Field, StrictFloat, StrictInt, confloat, conint, conlist
    pass # Add pass to avoid empty try/except if no imports are generated for this file.
except ImportError:
    # Pydantic v1
    from pydantic import BaseModel, Field, StrictFloat, StrictInt, confloat, conint, conlist
    pass # Add pass to avoid empty try/except if no imports are generated for this file.

class ObjectDetectionPrediction(BaseModel):
    """
    ObjectDetectionPrediction
    """
    category_id: StrictInt = Field(..., alias="categoryId", description="Category id of the prediction.")
    bbox: conlist(Union[StrictFloat, StrictInt]) = Field(..., description="Bounding box in (x, y, width, height) format where (x=0, y=0) is the top-left corner of the image.")
    score: Union[confloat(le=1, ge=0, strict=True), conint(le=1, ge=0, strict=True)] = Field(..., description="Detection confidence, range [0, 1].")
    probabilities: Optional[conlist(Union[confloat(ge=0, strict=True), conint(ge=0, strict=True)])] = Field(None, description="List with probability for each possible category (must sum to 1).")
    epsilon: Union[confloat(ge=0, strict=True), conint(ge=0, strict=True)] = Field(...)
    __properties = ["categoryId", "bbox", "score", "probabilities", "epsilon"]

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
    def from_json(cls, json_str: str) -> ObjectDetectionPrediction:
        """Create an instance of ObjectDetectionPrediction from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # set to None if probabilities (nullable) is None
        # and __fields_set__ contains the field
        if self.probabilities is None and "probabilities" in self.__fields_set__:
            _dict['probabilities' if by_alias else 'probabilities'] = None

        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> ObjectDetectionPrediction:
        """Create an instance of ObjectDetectionPrediction from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return ObjectDetectionPrediction.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in ObjectDetectionPrediction) in the input: " + str(obj))

        _obj = ObjectDetectionPrediction.parse_obj({
            "category_id": obj.get("categoryId"),
            "bbox": obj.get("bbox"),
            "score": obj.get("score"),
            "probabilities": obj.get("probabilities"),
            "epsilon": obj.get("epsilon")
        })
        return _obj

