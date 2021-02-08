# coding: utf-8

# flake8: noqa

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    OpenAPI spec version: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


from __future__ import absolute_import

# import apis into sdk package
from swagger_client.api.annotations_api import AnnotationsApi
from swagger_client.api.auth_api import AuthApi
from swagger_client.api.datasets_api import DatasetsApi
from swagger_client.api.embeddings_api import EmbeddingsApi
from swagger_client.api.internal_api import InternalApi
from swagger_client.api.jobs_api import JobsApi
from swagger_client.api.mappings_api import MappingsApi
from swagger_client.api.samples_api import SamplesApi
from swagger_client.api.samplings_api import SamplingsApi
from swagger_client.api.tags_api import TagsApi
# import ApiClient
from swagger_client.api_client import ApiClient
from swagger_client.configuration import Configuration
# import models into sdk package
from swagger_client.models.annotation_data import AnnotationData
from swagger_client.models.annotation_meta_data import AnnotationMetaData
from swagger_client.models.annotation_offer_data import AnnotationOfferData
from swagger_client.models.annotation_state import AnnotationState
from swagger_client.models.api_error_code import ApiErrorCode
from swagger_client.models.api_error_response import ApiErrorResponse
from swagger_client.models.async_task_data import AsyncTaskData
from swagger_client.models.body import Body
from swagger_client.models.create_cf_bucket_activity_request import CreateCFBucketActivityRequest
from swagger_client.models.create_entity_response import CreateEntityResponse
from swagger_client.models.dataset_data import DatasetData
from swagger_client.models.dataset_type import DatasetType
from swagger_client.models.embedding_data import EmbeddingData
from swagger_client.models.general_job_result import GeneralJobResult
from swagger_client.models.image_type import ImageType
from swagger_client.models.initial_tag_create_request import InitialTagCreateRequest
from swagger_client.models.inline_response200 import InlineResponse200
from swagger_client.models.job_result_type import JobResultType
from swagger_client.models.job_state import JobState
from swagger_client.models.job_status_data import JobStatusData
from swagger_client.models.job_status_data_result import JobStatusDataResult
from swagger_client.models.mongo_object_id import MongoObjectID
from swagger_client.models.object_id import ObjectId
from swagger_client.models.sample_data import SampleData
from swagger_client.models.sample_meta_data import SampleMetaData
from swagger_client.models.sampling_config import SamplingConfig
from swagger_client.models.sampling_config_stopping_condition import SamplingConfigStoppingCondition
from swagger_client.models.sampling_create_request import SamplingCreateRequest
from swagger_client.models.sampling_method import SamplingMethod
from swagger_client.models.tag_bit_mask_data import TagBitMaskData
from swagger_client.models.tag_change_data import TagChangeData
from swagger_client.models.tag_create_request import TagCreateRequest
from swagger_client.models.tag_data import TagData
from swagger_client.models.tag_filenames_data import TagFilenamesData
from swagger_client.models.tag_name import TagName
from swagger_client.models.timestamp import Timestamp
from swagger_client.models.write_csv_url_data import WriteCSVUrlData
