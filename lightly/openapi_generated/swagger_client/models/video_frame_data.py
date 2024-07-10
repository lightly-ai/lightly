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


from typing import Optional, Union
try:
    # Pydantic >=v1.10.17
    from pydantic.v1 import BaseModel, Field, StrictFloat, StrictInt, StrictStr
except ImportError:
    # Pydantic v1
    from pydantic import BaseModel, Field, StrictFloat, StrictInt, StrictStr

class VideoFrameData(BaseModel):
    """
    VideoFrameData
    """
    source_video: Optional[StrictStr] = Field(None, alias="sourceVideo", description="Name of the source video.")
    source_video_frame_index: Optional[Union[StrictFloat, StrictInt]] = Field(None, alias="sourceVideoFrameIndex", description="Index of the frame in the source video.")
    source_video_frame_timestamp: Optional[Union[StrictFloat, StrictInt]] = Field(None, alias="sourceVideoFrameTimestamp", description="Timestamp of the frame in the source video.")
    __properties = ["sourceVideo", "sourceVideoFrameIndex", "sourceVideoFrameTimestamp"]

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
    def from_json(cls, json_str: str) -> VideoFrameData:
        """Create an instance of VideoFrameData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> VideoFrameData:
        """Create an instance of VideoFrameData from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return VideoFrameData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in VideoFrameData) in the input: " + str(obj))

        _obj = VideoFrameData.parse_obj({
            "source_video": obj.get("sourceVideo"),
            "source_video_frame_index": obj.get("sourceVideoFrameIndex"),
            "source_video_frame_timestamp": obj.get("sourceVideoFrameTimestamp")
        })
        return _obj

