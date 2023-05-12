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



from typing import Optional
from pydantic import Extra,  BaseModel
from lightly.openapi_client.models.docker_worker_config_v2_lightly_collate import DockerWorkerConfigV2LightlyCollate
from lightly.openapi_client.models.docker_worker_config_v2_lightly_model import DockerWorkerConfigV2LightlyModel
from lightly.openapi_client.models.docker_worker_config_v2_lightly_trainer import DockerWorkerConfigV2LightlyTrainer
from lightly.openapi_client.models.docker_worker_config_v3_lightly_criterion import DockerWorkerConfigV3LightlyCriterion
from lightly.openapi_client.models.docker_worker_config_v3_lightly_loader import DockerWorkerConfigV3LightlyLoader
from lightly.openapi_client.models.docker_worker_config_v3_lightly_optimizer import DockerWorkerConfigV3LightlyOptimizer

class DockerWorkerConfigV2Lightly(BaseModel):
    """
    Lightly configurations which are passed to a Lightly Worker run. For information about the options see https://docs.lightly.ai/docs/all-configuration-options#run-configuration. 
    """
    loader: Optional[DockerWorkerConfigV3LightlyLoader] = None
    model: Optional[DockerWorkerConfigV2LightlyModel] = None
    trainer: Optional[DockerWorkerConfigV2LightlyTrainer] = None
    criterion: Optional[DockerWorkerConfigV3LightlyCriterion] = None
    optimizer: Optional[DockerWorkerConfigV3LightlyOptimizer] = None
    collate: Optional[DockerWorkerConfigV2LightlyCollate] = None
    __properties = ["loader", "model", "trainer", "criterion", "optimizer", "collate"]

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
    def from_json(cls, json_str: str) -> DockerWorkerConfigV2Lightly:
        """Create an instance of DockerWorkerConfigV2Lightly from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        """Returns the dictionary representation of the model using alias"""
        _dict = self.dict(by_alias=True,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of loader
        if self.loader:
            _dict['loader'] = self.loader.to_dict()
        # override the default output from pydantic by calling `to_dict()` of model
        if self.model:
            _dict['model'] = self.model.to_dict()
        # override the default output from pydantic by calling `to_dict()` of trainer
        if self.trainer:
            _dict['trainer'] = self.trainer.to_dict()
        # override the default output from pydantic by calling `to_dict()` of criterion
        if self.criterion:
            _dict['criterion'] = self.criterion.to_dict()
        # override the default output from pydantic by calling `to_dict()` of optimizer
        if self.optimizer:
            _dict['optimizer'] = self.optimizer.to_dict()
        # override the default output from pydantic by calling `to_dict()` of collate
        if self.collate:
            _dict['collate'] = self.collate.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DockerWorkerConfigV2Lightly:
        """Create an instance of DockerWorkerConfigV2Lightly from a dict"""
        if obj is None:
            return None

        if type(obj) is not dict:
            return DockerWorkerConfigV2Lightly.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DockerWorkerConfigV2Lightly) in the input: " + str(obj))

        _obj = DockerWorkerConfigV2Lightly.parse_obj({
            "loader": DockerWorkerConfigV3LightlyLoader.from_dict(obj.get("loader")) if obj.get("loader") is not None else None,
            "model": DockerWorkerConfigV2LightlyModel.from_dict(obj.get("model")) if obj.get("model") is not None else None,
            "trainer": DockerWorkerConfigV2LightlyTrainer.from_dict(obj.get("trainer")) if obj.get("trainer") is not None else None,
            "criterion": DockerWorkerConfigV3LightlyCriterion.from_dict(obj.get("criterion")) if obj.get("criterion") is not None else None,
            "optimizer": DockerWorkerConfigV3LightlyOptimizer.from_dict(obj.get("optimizer")) if obj.get("optimizer") is not None else None,
            "collate": DockerWorkerConfigV2LightlyCollate.from_dict(obj.get("collate")) if obj.get("collate") is not None else None
        })
        return _obj


