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


from typing import Dict, Optional
try:
    # Pydantic >=v1.10.17
    from pydantic.v1 import BaseModel, Field
except ImportError:
    # Pydantic v1
    from pydantic import BaseModel, Field
from lightly.openapi_generated.swagger_client.models.dataset_analysis import DatasetAnalysis
from lightly.openapi_generated.swagger_client.models.general_information import GeneralInformation
from lightly.openapi_generated.swagger_client.models.prediction_task_information import PredictionTaskInformation

class ReportV2(BaseModel):
    """
    ReportV2
    """
    general_information: Optional[GeneralInformation] = Field(None, alias="generalInformation")
    dataset_analysis: Optional[DatasetAnalysis] = Field(None, alias="datasetAnalysis")
    prediction_task_information: Optional[Dict[str, PredictionTaskInformation]] = Field(None, alias="predictionTaskInformation")
    __properties = ["generalInformation", "datasetAnalysis", "predictionTaskInformation"]

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
    def from_json(cls, json_str: str) -> ReportV2:
        """Create an instance of ReportV2 from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of general_information
        if self.general_information:
            _dict['generalInformation' if by_alias else 'general_information'] = self.general_information.to_dict(by_alias=by_alias)
        # override the default output from pydantic by calling `to_dict()` of dataset_analysis
        if self.dataset_analysis:
            _dict['datasetAnalysis' if by_alias else 'dataset_analysis'] = self.dataset_analysis.to_dict(by_alias=by_alias)
        # override the default output from pydantic by calling `to_dict()` of each value in prediction_task_information (dict)
        _field_dict = {}
        if self.prediction_task_information:
            for _key in self.prediction_task_information:
                if self.prediction_task_information[_key]:
                    _field_dict[_key] = self.prediction_task_information[_key].to_dict(by_alias=by_alias)
            _dict['predictionTaskInformation' if by_alias else 'prediction_task_information'] = _field_dict
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> ReportV2:
        """Create an instance of ReportV2 from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return ReportV2.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in ReportV2) in the input: " + str(obj))

        _obj = ReportV2.parse_obj({
            "general_information": GeneralInformation.from_dict(obj.get("generalInformation")) if obj.get("generalInformation") is not None else None,
            "dataset_analysis": DatasetAnalysis.from_dict(obj.get("datasetAnalysis")) if obj.get("datasetAnalysis") is not None else None,
            "prediction_task_information": dict(
                (_k, PredictionTaskInformation.from_dict(_v))
                for _k, _v in obj.get("predictionTaskInformation").items()
            )
            if obj.get("predictionTaskInformation") is not None
            else None
        })
        return _obj

