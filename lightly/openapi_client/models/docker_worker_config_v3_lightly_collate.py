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



from typing import List, Optional, Union
from pydantic import Extra,  BaseModel, Field, StrictFloat, StrictInt, confloat, conint, conlist

class DockerWorkerConfigV3LightlyCollate(BaseModel):
    """
    DockerWorkerConfigV3LightlyCollate
    """
    input_size: Optional[conint(strict=True, ge=1)] = Field(None, alias="inputSize")
    cj_prob: Optional[Union[confloat(le=1.0, ge=0.0, strict=True), conint(le=1, ge=0, strict=True)]] = Field(None, alias="cjProb")
    cj_bright: Optional[Union[confloat(le=1.0, ge=0.0, strict=True), conint(le=1, ge=0, strict=True)]] = Field(None, alias="cjBright")
    cj_contrast: Optional[Union[confloat(le=1.0, ge=0.0, strict=True), conint(le=1, ge=0, strict=True)]] = Field(None, alias="cjContrast")
    cj_sat: Optional[Union[confloat(le=1.0, ge=0.0, strict=True), conint(le=1, ge=0, strict=True)]] = Field(None, alias="cjSat")
    cj_hue: Optional[Union[confloat(le=1.0, ge=0.0, strict=True), conint(le=1, ge=0, strict=True)]] = Field(None, alias="cjHue")
    min_scale: Optional[Union[confloat(le=1.0, ge=0.0, strict=True), conint(le=1, ge=0, strict=True)]] = Field(None, alias="minScale")
    random_gray_scale: Optional[Union[confloat(le=1.0, ge=0.0, strict=True), conint(le=1, ge=0, strict=True)]] = Field(None, alias="randomGrayScale")
    gaussian_blur: Optional[Union[confloat(le=1.0, ge=0.0, strict=True), conint(le=1, ge=0, strict=True)]] = Field(None, alias="gaussianBlur")
    kernel_size: Optional[Union[confloat(ge=0.0, strict=True), conint(ge=0, strict=True)]] = Field(None, alias="kernelSize")
    sigmas: Optional[conlist(Union[confloat(gt=0, strict=True), conint(gt=0, strict=True)], max_items=2, min_items=2)] = None
    vf_prob: Optional[Union[confloat(le=1.0, ge=0.0, strict=True), conint(le=1, ge=0, strict=True)]] = Field(None, alias="vfProb")
    hf_prob: Optional[Union[confloat(le=1.0, ge=0.0, strict=True), conint(le=1, ge=0, strict=True)]] = Field(None, alias="hfProb")
    rr_prob: Optional[Union[confloat(le=1.0, ge=0.0, strict=True), conint(le=1, ge=0, strict=True)]] = Field(None, alias="rrProb")
    rr_degrees: Optional[conlist(Union[StrictFloat, StrictInt], max_items=2, min_items=2)] = Field(None, alias="rrDegrees")
    __properties = ["inputSize", "cjProb", "cjBright", "cjContrast", "cjSat", "cjHue", "minScale", "randomGrayScale", "gaussianBlur", "kernelSize", "sigmas", "vfProb", "hfProb", "rrProb", "rrDegrees"]

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
    def from_json(cls, json_str: str) -> DockerWorkerConfigV3LightlyCollate:
        """Create an instance of DockerWorkerConfigV3LightlyCollate from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # set to None if rr_degrees (nullable) is None
        # and __fields_set__ contains the field
        if self.rr_degrees is None and "rr_degrees" in self.__fields_set__:
            _dict['rrDegrees'] = None

        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DockerWorkerConfigV3LightlyCollate:
        """Create an instance of DockerWorkerConfigV3LightlyCollate from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return DockerWorkerConfigV3LightlyCollate.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DockerWorkerConfigV3LightlyCollate) in the input: " + str(obj))

        _obj = DockerWorkerConfigV3LightlyCollate.parse_obj({
            "input_size": obj.get("inputSize"),
            "cj_prob": obj.get("cjProb"),
            "cj_bright": obj.get("cjBright"),
            "cj_contrast": obj.get("cjContrast"),
            "cj_sat": obj.get("cjSat"),
            "cj_hue": obj.get("cjHue"),
            "min_scale": obj.get("minScale"),
            "random_gray_scale": obj.get("randomGrayScale"),
            "gaussian_blur": obj.get("gaussianBlur"),
            "kernel_size": obj.get("kernelSize"),
            "sigmas": obj.get("sigmas"),
            "vf_prob": obj.get("vfProb"),
            "hf_prob": obj.get("hfProb"),
            "rr_prob": obj.get("rrProb"),
            "rr_degrees": obj.get("rrDegrees")
        })
        return _obj


