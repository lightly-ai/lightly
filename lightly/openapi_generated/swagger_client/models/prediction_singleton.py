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
from lightly.openapi_generated.swagger_client.models.prediction_singleton_classification import PredictionSingletonClassification
from lightly.openapi_generated.swagger_client.models.prediction_singleton_instance_segmentation import PredictionSingletonInstanceSegmentation
from lightly.openapi_generated.swagger_client.models.prediction_singleton_keypoint_detection import PredictionSingletonKeypointDetection
from lightly.openapi_generated.swagger_client.models.prediction_singleton_object_detection import PredictionSingletonObjectDetection
from lightly.openapi_generated.swagger_client.models.prediction_singleton_semantic_segmentation import PredictionSingletonSemanticSegmentation
try:
    # Pydantic >=v1.10.17
    from pydantic.v1 import StrictStr, Field
except ImportError:
    # Pydantic v1
    from pydantic import StrictStr, Field

PREDICTIONSINGLETON_ONE_OF_SCHEMAS = ["PredictionSingletonClassification", "PredictionSingletonInstanceSegmentation", "PredictionSingletonKeypointDetection", "PredictionSingletonObjectDetection", "PredictionSingletonSemanticSegmentation"]

class PredictionSingleton(BaseModel):
    """
    PredictionSingleton
    """
    # data type: PredictionSingletonClassification
    oneof_schema_1_validator: Optional[PredictionSingletonClassification] = None
    # data type: PredictionSingletonObjectDetection
    oneof_schema_2_validator: Optional[PredictionSingletonObjectDetection] = None
    # data type: PredictionSingletonSemanticSegmentation
    oneof_schema_3_validator: Optional[PredictionSingletonSemanticSegmentation] = None
    # data type: PredictionSingletonInstanceSegmentation
    oneof_schema_4_validator: Optional[PredictionSingletonInstanceSegmentation] = None
    # data type: PredictionSingletonKeypointDetection
    oneof_schema_5_validator: Optional[PredictionSingletonKeypointDetection] = None
    actual_instance: Any
    one_of_schemas: List[str] = Field(PREDICTIONSINGLETON_ONE_OF_SCHEMAS, const=True)

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
        instance = PredictionSingleton.construct()
        error_messages = []
        match = 0
        # validate data type: PredictionSingletonClassification
        if not isinstance(v, PredictionSingletonClassification):
            error_messages.append(f"Error! Input type `{type(v)}` is not `PredictionSingletonClassification`")
        else:
            match += 1
        # validate data type: PredictionSingletonObjectDetection
        if not isinstance(v, PredictionSingletonObjectDetection):
            error_messages.append(f"Error! Input type `{type(v)}` is not `PredictionSingletonObjectDetection`")
        else:
            match += 1
        # validate data type: PredictionSingletonSemanticSegmentation
        if not isinstance(v, PredictionSingletonSemanticSegmentation):
            error_messages.append(f"Error! Input type `{type(v)}` is not `PredictionSingletonSemanticSegmentation`")
        else:
            match += 1
        # validate data type: PredictionSingletonInstanceSegmentation
        if not isinstance(v, PredictionSingletonInstanceSegmentation):
            error_messages.append(f"Error! Input type `{type(v)}` is not `PredictionSingletonInstanceSegmentation`")
        else:
            match += 1
        # validate data type: PredictionSingletonKeypointDetection
        if not isinstance(v, PredictionSingletonKeypointDetection):
            error_messages.append(f"Error! Input type `{type(v)}` is not `PredictionSingletonKeypointDetection`")
        else:
            match += 1
        if match > 1:
            # more than 1 match
            raise ValueError("Multiple matches found when setting `actual_instance` in PredictionSingleton with oneOf schemas: PredictionSingletonClassification, PredictionSingletonInstanceSegmentation, PredictionSingletonKeypointDetection, PredictionSingletonObjectDetection, PredictionSingletonSemanticSegmentation. Details: " + ", ".join(error_messages))
        elif match == 0:
            # no match
            raise ValueError("No match found when setting `actual_instance` in PredictionSingleton with oneOf schemas: PredictionSingletonClassification, PredictionSingletonInstanceSegmentation, PredictionSingletonKeypointDetection, PredictionSingletonObjectDetection, PredictionSingletonSemanticSegmentation. Details: " + ", ".join(error_messages))
        else:
            return v

    @classmethod
    def from_dict(cls, obj: dict) -> PredictionSingleton:
        return cls.from_json(json.dumps(obj))

    @classmethod
    def from_json(cls, json_str: str) -> PredictionSingleton:
        """Returns the object represented by the json string"""
        instance = PredictionSingleton.construct()
        error_messages = []
        match = 0

        # use oneOf discriminator to lookup the data type
        _data_type = json.loads(json_str).get("type")
        if not _data_type:
            raise ValueError("Failed to lookup data type from the field `type` in the input.")

        # check if data type is `PredictionSingletonClassification`
        if _data_type == "CLASSIFICATION":
            instance.actual_instance = PredictionSingletonClassification.from_json(json_str)
            return instance

        # check if data type is `PredictionSingletonInstanceSegmentation`
        if _data_type == "INSTANCE_SEGMENTATION":
            instance.actual_instance = PredictionSingletonInstanceSegmentation.from_json(json_str)
            return instance

        # check if data type is `PredictionSingletonKeypointDetection`
        if _data_type == "KEYPOINT_DETECTION":
            instance.actual_instance = PredictionSingletonKeypointDetection.from_json(json_str)
            return instance

        # check if data type is `PredictionSingletonObjectDetection`
        if _data_type == "OBJECT_DETECTION":
            instance.actual_instance = PredictionSingletonObjectDetection.from_json(json_str)
            return instance

        # check if data type is `PredictionSingletonClassification`
        if _data_type == "PredictionSingletonClassification":
            instance.actual_instance = PredictionSingletonClassification.from_json(json_str)
            return instance

        # check if data type is `PredictionSingletonInstanceSegmentation`
        if _data_type == "PredictionSingletonInstanceSegmentation":
            instance.actual_instance = PredictionSingletonInstanceSegmentation.from_json(json_str)
            return instance

        # check if data type is `PredictionSingletonKeypointDetection`
        if _data_type == "PredictionSingletonKeypointDetection":
            instance.actual_instance = PredictionSingletonKeypointDetection.from_json(json_str)
            return instance

        # check if data type is `PredictionSingletonObjectDetection`
        if _data_type == "PredictionSingletonObjectDetection":
            instance.actual_instance = PredictionSingletonObjectDetection.from_json(json_str)
            return instance

        # check if data type is `PredictionSingletonSemanticSegmentation`
        if _data_type == "PredictionSingletonSemanticSegmentation":
            instance.actual_instance = PredictionSingletonSemanticSegmentation.from_json(json_str)
            return instance

        # check if data type is `PredictionSingletonSemanticSegmentation`
        if _data_type == "SEMANTIC_SEGMENTATION":
            instance.actual_instance = PredictionSingletonSemanticSegmentation.from_json(json_str)
            return instance

        # deserialize data into PredictionSingletonClassification
        try:
            instance.actual_instance = PredictionSingletonClassification.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into PredictionSingletonObjectDetection
        try:
            instance.actual_instance = PredictionSingletonObjectDetection.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into PredictionSingletonSemanticSegmentation
        try:
            instance.actual_instance = PredictionSingletonSemanticSegmentation.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into PredictionSingletonInstanceSegmentation
        try:
            instance.actual_instance = PredictionSingletonInstanceSegmentation.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))
        # deserialize data into PredictionSingletonKeypointDetection
        try:
            instance.actual_instance = PredictionSingletonKeypointDetection.from_json(json_str)
            match += 1
        except (ValidationError, ValueError) as e:
            error_messages.append(str(e))

        if match > 1:
            # more than 1 match
            raise ValueError("Multiple matches found when deserializing the JSON string into PredictionSingleton with oneOf schemas: PredictionSingletonClassification, PredictionSingletonInstanceSegmentation, PredictionSingletonKeypointDetection, PredictionSingletonObjectDetection, PredictionSingletonSemanticSegmentation. Details: " + ", ".join(error_messages))
        elif match == 0:
            # no match
            raise ValueError("No match found when deserializing the JSON string into PredictionSingleton with oneOf schemas: PredictionSingletonClassification, PredictionSingletonInstanceSegmentation, PredictionSingletonKeypointDetection, PredictionSingletonObjectDetection, PredictionSingletonSemanticSegmentation. Details: " + ", ".join(error_messages))
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

