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
from lightly.openapi_generated.swagger_client.api.collaboration_api import CollaborationApi
from lightly.openapi_generated.swagger_client.api.datasets_api import DatasetsApi
from lightly.openapi_generated.swagger_client.api.datasources_api import DatasourcesApi
from lightly.openapi_generated.swagger_client.api.docker_api import DockerApi
from lightly.openapi_generated.swagger_client.api.embeddings_api import EmbeddingsApi
from lightly.openapi_generated.swagger_client.api.embeddings2d_api import Embeddings2dApi
from lightly.openapi_generated.swagger_client.api.jobs_api import JobsApi
from lightly.openapi_generated.swagger_client.api.mappings_api import MappingsApi
from lightly.openapi_generated.swagger_client.api.meta_data_configurations_api import MetaDataConfigurationsApi
from lightly.openapi_generated.swagger_client.api.predictions_api import PredictionsApi
from lightly.openapi_generated.swagger_client.api.quota_api import QuotaApi
from lightly.openapi_generated.swagger_client.api.samples_api import SamplesApi
from lightly.openapi_generated.swagger_client.api.samplings_api import SamplingsApi
from lightly.openapi_generated.swagger_client.api.scores_api import ScoresApi
from lightly.openapi_generated.swagger_client.api.tags_api import TagsApi
from lightly.openapi_generated.swagger_client.api.versioning_api import VersioningApi

# import ApiClient
from lightly.openapi_generated.swagger_client.api_client import ApiClient
from lightly.openapi_generated.swagger_client.configuration import Configuration
# import models into sdk package
from lightly.openapi_generated.swagger_client.models.access_role import AccessRole
from lightly.openapi_generated.swagger_client.models.active_learning_score_create_request import ActiveLearningScoreCreateRequest
from lightly.openapi_generated.swagger_client.models.active_learning_score_data import ActiveLearningScoreData
from lightly.openapi_generated.swagger_client.models.active_learning_score_type import ActiveLearningScoreType
from lightly.openapi_generated.swagger_client.models.active_learning_scores import ActiveLearningScores
from lightly.openapi_generated.swagger_client.models.api_error_code import ApiErrorCode
from lightly.openapi_generated.swagger_client.models.api_error_response import ApiErrorResponse
from lightly.openapi_generated.swagger_client.models.async_task_data import AsyncTaskData
from lightly.openapi_generated.swagger_client.models.category_id import CategoryId
from lightly.openapi_generated.swagger_client.models.category_name import CategoryName
from lightly.openapi_generated.swagger_client.models.configuration_data import ConfigurationData
from lightly.openapi_generated.swagger_client.models.configuration_entry import ConfigurationEntry
from lightly.openapi_generated.swagger_client.models.configuration_set_request import ConfigurationSetRequest
from lightly.openapi_generated.swagger_client.models.configuration_value_data_type import ConfigurationValueDataType
from lightly.openapi_generated.swagger_client.models.create_docker_worker_registry_entry_request import CreateDockerWorkerRegistryEntryRequest
from lightly.openapi_generated.swagger_client.models.create_entity_response import CreateEntityResponse
from lightly.openapi_generated.swagger_client.models.crop_data import CropData
from lightly.openapi_generated.swagger_client.models.custom_sample_meta_data import CustomSampleMetaData
from lightly.openapi_generated.swagger_client.models.dataset_create_request import DatasetCreateRequest
from lightly.openapi_generated.swagger_client.models.dataset_data import DatasetData
from lightly.openapi_generated.swagger_client.models.dataset_data_enriched import DatasetDataEnriched
from lightly.openapi_generated.swagger_client.models.dataset_embedding_data import DatasetEmbeddingData
from lightly.openapi_generated.swagger_client.models.dataset_name import DatasetName
from lightly.openapi_generated.swagger_client.models.dataset_type import DatasetType
from lightly.openapi_generated.swagger_client.models.dataset_update_request import DatasetUpdateRequest
from lightly.openapi_generated.swagger_client.models.datasource_config import DatasourceConfig
from lightly.openapi_generated.swagger_client.models.datasource_config_azure import DatasourceConfigAzure
from lightly.openapi_generated.swagger_client.models.datasource_config_base import DatasourceConfigBase
from lightly.openapi_generated.swagger_client.models.datasource_config_gcs import DatasourceConfigGCS
from lightly.openapi_generated.swagger_client.models.datasource_config_lightly import DatasourceConfigLIGHTLY
from lightly.openapi_generated.swagger_client.models.datasource_config_local import DatasourceConfigLOCAL
from lightly.openapi_generated.swagger_client.models.datasource_config_obs import DatasourceConfigOBS
from lightly.openapi_generated.swagger_client.models.datasource_config_s3 import DatasourceConfigS3
from lightly.openapi_generated.swagger_client.models.datasource_config_s3_delegated_access import DatasourceConfigS3DelegatedAccess
from lightly.openapi_generated.swagger_client.models.datasource_config_verify_data import DatasourceConfigVerifyData
from lightly.openapi_generated.swagger_client.models.datasource_processed_until_timestamp_request import DatasourceProcessedUntilTimestampRequest
from lightly.openapi_generated.swagger_client.models.datasource_processed_until_timestamp_response import DatasourceProcessedUntilTimestampResponse
from lightly.openapi_generated.swagger_client.models.datasource_purpose import DatasourcePurpose
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data import DatasourceRawSamplesData
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data_row import DatasourceRawSamplesDataRow
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_metadata_data import DatasourceRawSamplesMetadataData
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_metadata_data_row import DatasourceRawSamplesMetadataDataRow
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_predictions_data import DatasourceRawSamplesPredictionsData
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_predictions_data_row import DatasourceRawSamplesPredictionsDataRow
from lightly.openapi_generated.swagger_client.models.dimensionality_reduction_method import DimensionalityReductionMethod
from lightly.openapi_generated.swagger_client.models.docker_license_information import DockerLicenseInformation
from lightly.openapi_generated.swagger_client.models.docker_run_create_request import DockerRunCreateRequest
from lightly.openapi_generated.swagger_client.models.docker_run_data import DockerRunData
from lightly.openapi_generated.swagger_client.models.docker_run_data_log import DockerRunDataLog
from lightly.openapi_generated.swagger_client.models.docker_run_scheduled_create_request import DockerRunScheduledCreateRequest
from lightly.openapi_generated.swagger_client.models.docker_run_scheduled_data import DockerRunScheduledData
from lightly.openapi_generated.swagger_client.models.docker_run_scheduled_priority import DockerRunScheduledPriority
from lightly.openapi_generated.swagger_client.models.docker_run_scheduled_state import DockerRunScheduledState
from lightly.openapi_generated.swagger_client.models.docker_run_scheduled_update_request import DockerRunScheduledUpdateRequest
from lightly.openapi_generated.swagger_client.models.docker_run_state import DockerRunState
from lightly.openapi_generated.swagger_client.models.docker_run_update_request import DockerRunUpdateRequest
from lightly.openapi_generated.swagger_client.models.docker_task_description import DockerTaskDescription
from lightly.openapi_generated.swagger_client.models.docker_user_stats import DockerUserStats
from lightly.openapi_generated.swagger_client.models.docker_worker_config import DockerWorkerConfig
from lightly.openapi_generated.swagger_client.models.docker_worker_config_create_request import DockerWorkerConfigCreateRequest
from lightly.openapi_generated.swagger_client.models.docker_worker_config_data import DockerWorkerConfigData
from lightly.openapi_generated.swagger_client.models.docker_worker_name import DockerWorkerName
from lightly.openapi_generated.swagger_client.models.docker_worker_registry_entry_data import DockerWorkerRegistryEntryData
from lightly.openapi_generated.swagger_client.models.docker_worker_state import DockerWorkerState
from lightly.openapi_generated.swagger_client.models.docker_worker_type import DockerWorkerType
from lightly.openapi_generated.swagger_client.models.embedding2d_coordinates import Embedding2dCoordinates
from lightly.openapi_generated.swagger_client.models.embedding2d_create_request import Embedding2dCreateRequest
from lightly.openapi_generated.swagger_client.models.embedding2d_data import Embedding2dData
from lightly.openapi_generated.swagger_client.models.embedding_data import EmbeddingData
from lightly.openapi_generated.swagger_client.models.embedding_id_is_processed_body import EmbeddingIdIsProcessedBody
from lightly.openapi_generated.swagger_client.models.file_name_format import FileNameFormat
from lightly.openapi_generated.swagger_client.models.file_output_format import FileOutputFormat
from lightly.openapi_generated.swagger_client.models.filename_and_read_url import FilenameAndReadUrl
from lightly.openapi_generated.swagger_client.models.filename_and_read_urls import FilenameAndReadUrls
from lightly.openapi_generated.swagger_client.models.general_job_result import GeneralJobResult
from lightly.openapi_generated.swagger_client.models.image_type import ImageType
from lightly.openapi_generated.swagger_client.models.initial_tag_create_request import InitialTagCreateRequest
from lightly.openapi_generated.swagger_client.models.job_result_type import JobResultType
from lightly.openapi_generated.swagger_client.models.job_state import JobState
from lightly.openapi_generated.swagger_client.models.job_status_data import JobStatusData
from lightly.openapi_generated.swagger_client.models.job_status_data_result import JobStatusDataResult
from lightly.openapi_generated.swagger_client.models.job_status_meta import JobStatusMeta
from lightly.openapi_generated.swagger_client.models.job_status_upload_method import JobStatusUploadMethod
from lightly.openapi_generated.swagger_client.models.jobs_data import JobsData
from lightly.openapi_generated.swagger_client.models.label_box_data_row import LabelBoxDataRow
from lightly.openapi_generated.swagger_client.models.label_box_data_rows import LabelBoxDataRows
from lightly.openapi_generated.swagger_client.models.label_studio_task import LabelStudioTask
from lightly.openapi_generated.swagger_client.models.label_studio_task_data import LabelStudioTaskData
from lightly.openapi_generated.swagger_client.models.label_studio_tasks import LabelStudioTasks
from lightly.openapi_generated.swagger_client.models.mongo_object_id import MongoObjectID
from lightly.openapi_generated.swagger_client.models.object_id import ObjectId
from lightly.openapi_generated.swagger_client.models.path_safe_name import PathSafeName
from lightly.openapi_generated.swagger_client.models.prediction_singleton import PredictionSingleton
from lightly.openapi_generated.swagger_client.models.prediction_singleton_base import PredictionSingletonBase
from lightly.openapi_generated.swagger_client.models.prediction_singleton_classification import PredictionSingletonClassification
from lightly.openapi_generated.swagger_client.models.prediction_singleton_instance_segmentation import PredictionSingletonInstanceSegmentation
from lightly.openapi_generated.swagger_client.models.prediction_singleton_keypoint_detection import PredictionSingletonKeypointDetection
from lightly.openapi_generated.swagger_client.models.prediction_singleton_object_detection import PredictionSingletonObjectDetection
from lightly.openapi_generated.swagger_client.models.prediction_singletons import PredictionSingletons
from lightly.openapi_generated.swagger_client.models.prediction_task_schema import PredictionTaskSchema
from lightly.openapi_generated.swagger_client.models.prediction_task_schema_category import PredictionTaskSchemaCategory
from lightly.openapi_generated.swagger_client.models.probabilities import Probabilities
from lightly.openapi_generated.swagger_client.models.read_url import ReadUrl
from lightly.openapi_generated.swagger_client.models.redirected_read_url import RedirectedReadUrl
from lightly.openapi_generated.swagger_client.models.s3_region import S3Region
from lightly.openapi_generated.swagger_client.models.s3_server_side_encryption_kms_key import S3ServerSideEncryptionKMSKey
from lightly.openapi_generated.swagger_client.models.sama_task import SamaTask
from lightly.openapi_generated.swagger_client.models.sama_task_data import SamaTaskData
from lightly.openapi_generated.swagger_client.models.sama_tasks import SamaTasks
from lightly.openapi_generated.swagger_client.models.sample_create_request import SampleCreateRequest
from lightly.openapi_generated.swagger_client.models.sample_data import SampleData
from lightly.openapi_generated.swagger_client.models.sample_data_modes import SampleDataModes
from lightly.openapi_generated.swagger_client.models.sample_meta_data import SampleMetaData
from lightly.openapi_generated.swagger_client.models.sample_partial_mode import SamplePartialMode
from lightly.openapi_generated.swagger_client.models.sample_type import SampleType
from lightly.openapi_generated.swagger_client.models.sample_update_request import SampleUpdateRequest
from lightly.openapi_generated.swagger_client.models.sample_write_urls import SampleWriteUrls
from lightly.openapi_generated.swagger_client.models.sampling_config import SamplingConfig
from lightly.openapi_generated.swagger_client.models.sampling_config_stopping_condition import SamplingConfigStoppingCondition
from lightly.openapi_generated.swagger_client.models.sampling_create_request import SamplingCreateRequest
from lightly.openapi_generated.swagger_client.models.sampling_method import SamplingMethod
from lightly.openapi_generated.swagger_client.models.score import Score
from lightly.openapi_generated.swagger_client.models.selection_config import SelectionConfig
from lightly.openapi_generated.swagger_client.models.selection_config_entry import SelectionConfigEntry
from lightly.openapi_generated.swagger_client.models.selection_config_entry_input import SelectionConfigEntryInput
from lightly.openapi_generated.swagger_client.models.selection_config_entry_strategy import SelectionConfigEntryStrategy
from lightly.openapi_generated.swagger_client.models.selection_input_predictions_name import SelectionInputPredictionsName
from lightly.openapi_generated.swagger_client.models.selection_input_type import SelectionInputType
from lightly.openapi_generated.swagger_client.models.selection_strategy_threshold_operation import SelectionStrategyThresholdOperation
from lightly.openapi_generated.swagger_client.models.selection_strategy_type import SelectionStrategyType
from lightly.openapi_generated.swagger_client.models.shared_access_config_create_request import SharedAccessConfigCreateRequest
from lightly.openapi_generated.swagger_client.models.shared_access_config_data import SharedAccessConfigData
from lightly.openapi_generated.swagger_client.models.shared_access_type import SharedAccessType
from lightly.openapi_generated.swagger_client.models.tag_active_learning_scores_data import TagActiveLearningScoresData
from lightly.openapi_generated.swagger_client.models.tag_arithmetics_operation import TagArithmeticsOperation
from lightly.openapi_generated.swagger_client.models.tag_arithmetics_request import TagArithmeticsRequest
from lightly.openapi_generated.swagger_client.models.tag_arithmetics_response import TagArithmeticsResponse
from lightly.openapi_generated.swagger_client.models.tag_bit_mask_data import TagBitMaskData
from lightly.openapi_generated.swagger_client.models.tag_bit_mask_response import TagBitMaskResponse
from lightly.openapi_generated.swagger_client.models.tag_change_data import TagChangeData
from lightly.openapi_generated.swagger_client.models.tag_create_request import TagCreateRequest
from lightly.openapi_generated.swagger_client.models.tag_creator import TagCreator
from lightly.openapi_generated.swagger_client.models.tag_data import TagData
from lightly.openapi_generated.swagger_client.models.tag_filenames_data import TagFilenamesData
from lightly.openapi_generated.swagger_client.models.tag_name import TagName
from lightly.openapi_generated.swagger_client.models.tag_update_request import TagUpdateRequest
from lightly.openapi_generated.swagger_client.models.tag_upsize_request import TagUpsizeRequest
from lightly.openapi_generated.swagger_client.models.task_name import TaskName
from lightly.openapi_generated.swagger_client.models.task_type import TaskType
from lightly.openapi_generated.swagger_client.models.timestamp import Timestamp
from lightly.openapi_generated.swagger_client.models.timestamp_seconds import TimestampSeconds
from lightly.openapi_generated.swagger_client.models.trigger2d_embedding_job_request import Trigger2dEmbeddingJobRequest
from lightly.openapi_generated.swagger_client.models.update_docker_worker_registry_entry_request import UpdateDockerWorkerRegistryEntryRequest
from lightly.openapi_generated.swagger_client.models.version_number import VersionNumber
from lightly.openapi_generated.swagger_client.models.video_frame_data import VideoFrameData
from lightly.openapi_generated.swagger_client.models.write_csv_url_data import WriteCSVUrlData
