# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://openapi-generator.tech
"""

import re  # noqa: F401
import sys  # noqa: F401
import typing  # noqa: F401

from frozendict import frozendict  # noqa: F401

import decimal  # noqa: F401
from datetime import date, datetime  # noqa: F401
from frozendict import frozendict  # noqa: F401

from lightly.openapi_generated.swagger_client.schemas import (  # noqa: F401
    AnyTypeSchema,
    ComposedSchema,
    DictSchema,
    ListSchema,
    StrSchema,
    IntSchema,
    Int32Schema,
    Int64Schema,
    Float32Schema,
    Float64Schema,
    NumberSchema,
    DateSchema,
    DateTimeSchema,
    DecimalSchema,
    BoolSchema,
    BinarySchema,
    NoneSchema,
    none_type,
    Configuration,
    Unset,
    unset,
    ComposedBase,
    ListBase,
    DictBase,
    NoneBase,
    StrBase,
    IntBase,
    Int32Base,
    Int64Base,
    Float32Base,
    Float64Base,
    NumberBase,
    DateBase,
    DateTimeBase,
    BoolBase,
    BinaryBase,
    Schema,
    _SchemaValidator,
    _SchemaTypeChecker,
    _SchemaEnumMaker
)


class SampleData(
    DictSchema
):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """
    _required_property_names = set((
        'id',
        'type',
        'fileName',
    ))

    @classmethod
    @property
    def id(cls) -> typing.Type['MongoObjectID']:
        return MongoObjectID

    @classmethod
    @property
    def type(cls) -> typing.Type['SampleType']:
        return SampleType

    @classmethod
    @property
    def datasetId(cls) -> typing.Type['MongoObjectID']:
        return MongoObjectID
    fileName = StrSchema
    
    
    class thumbName(
        _SchemaTypeChecker(typing.Union[none_type, str, ]),
        StrBase,
        NoneBase,
        Schema
    ):
    
        def __new__(
            cls,
            *args: typing.Union[str, None, ],
            _configuration: typing.Optional[Configuration] = None,
        ) -> 'thumbName':
            return super().__new__(
                cls,
                *args,
                _configuration=_configuration,
            )
    exif = DictSchema
    index = IntSchema

    @classmethod
    @property
    def createdAt(cls) -> typing.Type['Timestamp']:
        return Timestamp

    @classmethod
    @property
    def lastModifiedAt(cls) -> typing.Type['Timestamp']:
        return Timestamp

    @classmethod
    @property
    def metaData(cls) -> typing.Type['SampleMetaData']:
        return SampleMetaData

    @classmethod
    @property
    def customMetaData(cls) -> typing.Type['CustomSampleMetaData']:
        return CustomSampleMetaData

    @classmethod
    @property
    def videoFrameData(cls) -> typing.Type['VideoFrameData']:
        return VideoFrameData

    @classmethod
    @property
    def cropData(cls) -> typing.Type['CropData']:
        return CropData


    def __new__(
        cls,
        *args: typing.Union[dict, frozendict, ],
        id: id,
        type: type,
        fileName: fileName,
        datasetId: typing.Union['MongoObjectID', Unset] = unset,
        thumbName: typing.Union[thumbName, Unset] = unset,
        exif: typing.Union[exif, Unset] = unset,
        index: typing.Union[index, Unset] = unset,
        createdAt: typing.Union['Timestamp', Unset] = unset,
        lastModifiedAt: typing.Union['Timestamp', Unset] = unset,
        metaData: typing.Union['SampleMetaData', Unset] = unset,
        customMetaData: typing.Union['CustomSampleMetaData', Unset] = unset,
        videoFrameData: typing.Union['VideoFrameData', Unset] = unset,
        cropData: typing.Union['CropData', Unset] = unset,
        _configuration: typing.Optional[Configuration] = None,
        **kwargs: typing.Type[Schema],
    ) -> 'SampleData':
        return super().__new__(
            cls,
            *args,
            id=id,
            type=type,
            fileName=fileName,
            datasetId=datasetId,
            thumbName=thumbName,
            exif=exif,
            index=index,
            createdAt=createdAt,
            lastModifiedAt=lastModifiedAt,
            metaData=metaData,
            customMetaData=customMetaData,
            videoFrameData=videoFrameData,
            cropData=cropData,
            _configuration=_configuration,
            **kwargs,
        )

from lightly.openapi_generated.swagger_client.model.crop_data import CropData
from lightly.openapi_generated.swagger_client.model.custom_sample_meta_data import CustomSampleMetaData
from lightly.openapi_generated.swagger_client.model.mongo_object_id import MongoObjectID
from lightly.openapi_generated.swagger_client.model.sample_meta_data import SampleMetaData
from lightly.openapi_generated.swagger_client.model.sample_type import SampleType
from lightly.openapi_generated.swagger_client.model.timestamp import Timestamp
from lightly.openapi_generated.swagger_client.model.video_frame_data import VideoFrameData
