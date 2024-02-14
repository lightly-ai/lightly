# coding: utf-8

# flake8: noqa

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""


__version__ = "1.0.0"

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
from lightly.openapi_generated.swagger_client.api.teams_api import TeamsApi
from lightly.openapi_generated.swagger_client.api.versioning_api import VersioningApi

# import ApiClient
from lightly.openapi_generated.swagger_client.api_response import ApiResponse
from lightly.openapi_generated.swagger_client.api_client import ApiClient
from lightly.openapi_generated.swagger_client.configuration import Configuration
from lightly.openapi_generated.swagger_client.exceptions import OpenApiException
from lightly.openapi_generated.swagger_client.exceptions import ApiTypeError
from lightly.openapi_generated.swagger_client.exceptions import ApiValueError
from lightly.openapi_generated.swagger_client.exceptions import ApiKeyError
from lightly.openapi_generated.swagger_client.exceptions import ApiAttributeError
from lightly.openapi_generated.swagger_client.exceptions import ApiException

# import models into sdk package
from lightly.openapi_generated.swagger_client.models.active_learning_score_create_request import ActiveLearningScoreCreateRequest
from lightly.openapi_generated.swagger_client.models.active_learning_score_data import ActiveLearningScoreData
from lightly.openapi_generated.swagger_client.models.active_learning_score_types_v2_data import ActiveLearningScoreTypesV2Data
from lightly.openapi_generated.swagger_client.models.active_learning_score_v2_data import ActiveLearningScoreV2Data
from lightly.openapi_generated.swagger_client.models.api_error_code import ApiErrorCode
from lightly.openapi_generated.swagger_client.models.api_error_response import ApiErrorResponse
from lightly.openapi_generated.swagger_client.models.async_task_data import AsyncTaskData
from lightly.openapi_generated.swagger_client.models.configuration_data import ConfigurationData
from lightly.openapi_generated.swagger_client.models.configuration_entry import ConfigurationEntry
from lightly.openapi_generated.swagger_client.models.configuration_set_request import ConfigurationSetRequest
from lightly.openapi_generated.swagger_client.models.configuration_value_data_type import ConfigurationValueDataType
from lightly.openapi_generated.swagger_client.models.create_cf_bucket_activity_request import CreateCFBucketActivityRequest
from lightly.openapi_generated.swagger_client.models.create_docker_worker_registry_entry_request import CreateDockerWorkerRegistryEntryRequest
from lightly.openapi_generated.swagger_client.models.create_entity_response import CreateEntityResponse
from lightly.openapi_generated.swagger_client.models.create_sample_with_write_urls_response import CreateSampleWithWriteUrlsResponse
from lightly.openapi_generated.swagger_client.models.create_team_membership_request import CreateTeamMembershipRequest
from lightly.openapi_generated.swagger_client.models.creator import Creator
from lightly.openapi_generated.swagger_client.models.crop_data import CropData
from lightly.openapi_generated.swagger_client.models.dataset_create_request import DatasetCreateRequest
from lightly.openapi_generated.swagger_client.models.dataset_creator import DatasetCreator
from lightly.openapi_generated.swagger_client.models.dataset_data import DatasetData
from lightly.openapi_generated.swagger_client.models.dataset_data_enriched import DatasetDataEnriched
from lightly.openapi_generated.swagger_client.models.dataset_embedding_data import DatasetEmbeddingData
from lightly.openapi_generated.swagger_client.models.dataset_type import DatasetType
from lightly.openapi_generated.swagger_client.models.dataset_update_request import DatasetUpdateRequest
from lightly.openapi_generated.swagger_client.models.datasource_config import DatasourceConfig
from lightly.openapi_generated.swagger_client.models.datasource_config_azure import DatasourceConfigAzure
from lightly.openapi_generated.swagger_client.models.datasource_config_azure_all_of import DatasourceConfigAzureAllOf
from lightly.openapi_generated.swagger_client.models.datasource_config_base import DatasourceConfigBase
from lightly.openapi_generated.swagger_client.models.datasource_config_base_full_path import DatasourceConfigBaseFullPath
from lightly.openapi_generated.swagger_client.models.datasource_config_gcs import DatasourceConfigGCS
from lightly.openapi_generated.swagger_client.models.datasource_config_gcs_all_of import DatasourceConfigGCSAllOf
from lightly.openapi_generated.swagger_client.models.datasource_config_lightly import DatasourceConfigLIGHTLY
from lightly.openapi_generated.swagger_client.models.datasource_config_local import DatasourceConfigLOCAL
from lightly.openapi_generated.swagger_client.models.datasource_config_local_all_of import DatasourceConfigLOCALAllOf
from lightly.openapi_generated.swagger_client.models.datasource_config_obs import DatasourceConfigOBS
from lightly.openapi_generated.swagger_client.models.datasource_config_obs_all_of import DatasourceConfigOBSAllOf
from lightly.openapi_generated.swagger_client.models.datasource_config_s3 import DatasourceConfigS3
from lightly.openapi_generated.swagger_client.models.datasource_config_s3_all_of import DatasourceConfigS3AllOf
from lightly.openapi_generated.swagger_client.models.datasource_config_s3_delegated_access import DatasourceConfigS3DelegatedAccess
from lightly.openapi_generated.swagger_client.models.datasource_config_s3_delegated_access_all_of import DatasourceConfigS3DelegatedAccessAllOf
from lightly.openapi_generated.swagger_client.models.datasource_config_verify_data import DatasourceConfigVerifyData
from lightly.openapi_generated.swagger_client.models.datasource_config_verify_data_errors import DatasourceConfigVerifyDataErrors
from lightly.openapi_generated.swagger_client.models.datasource_processed_until_timestamp_request import DatasourceProcessedUntilTimestampRequest
from lightly.openapi_generated.swagger_client.models.datasource_processed_until_timestamp_response import DatasourceProcessedUntilTimestampResponse
from lightly.openapi_generated.swagger_client.models.datasource_purpose import DatasourcePurpose
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data import DatasourceRawSamplesData
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_data_row import DatasourceRawSamplesDataRow
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_metadata_data import DatasourceRawSamplesMetadataData
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_metadata_data_row import DatasourceRawSamplesMetadataDataRow
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_predictions_data import DatasourceRawSamplesPredictionsData
from lightly.openapi_generated.swagger_client.models.datasource_raw_samples_predictions_data_row import DatasourceRawSamplesPredictionsDataRow
from lightly.openapi_generated.swagger_client.models.delegated_access_external_ids_inner import DelegatedAccessExternalIdsInner
from lightly.openapi_generated.swagger_client.models.dimensionality_reduction_method import DimensionalityReductionMethod
from lightly.openapi_generated.swagger_client.models.docker_license_information import DockerLicenseInformation
from lightly.openapi_generated.swagger_client.models.docker_run_artifact_create_request import DockerRunArtifactCreateRequest
from lightly.openapi_generated.swagger_client.models.docker_run_artifact_created_data import DockerRunArtifactCreatedData
from lightly.openapi_generated.swagger_client.models.docker_run_artifact_data import DockerRunArtifactData
from lightly.openapi_generated.swagger_client.models.docker_run_artifact_storage_location import DockerRunArtifactStorageLocation
from lightly.openapi_generated.swagger_client.models.docker_run_artifact_type import DockerRunArtifactType
from lightly.openapi_generated.swagger_client.models.docker_run_create_request import DockerRunCreateRequest
from lightly.openapi_generated.swagger_client.models.docker_run_data import DockerRunData
from lightly.openapi_generated.swagger_client.models.docker_run_log_create_entry_data import DockerRunLogCreateEntryData
from lightly.openapi_generated.swagger_client.models.docker_run_log_data import DockerRunLogData
from lightly.openapi_generated.swagger_client.models.docker_run_log_docker_load import DockerRunLogDockerLoad
from lightly.openapi_generated.swagger_client.models.docker_run_log_entry_data import DockerRunLogEntryData
from lightly.openapi_generated.swagger_client.models.docker_run_log_entry_data_base import DockerRunLogEntryDataBase
from lightly.openapi_generated.swagger_client.models.docker_run_log_level import DockerRunLogLevel
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
from lightly.openapi_generated.swagger_client.models.docker_worker_config_omni_v2_create_request import DockerWorkerConfigOmniV2CreateRequest
from lightly.openapi_generated.swagger_client.models.docker_worker_config_omni_v2_create_request_all_of import DockerWorkerConfigOmniV2CreateRequestAllOf
from lightly.openapi_generated.swagger_client.models.docker_worker_config_omni_v3_create_request import DockerWorkerConfigOmniV3CreateRequest
from lightly.openapi_generated.swagger_client.models.docker_worker_config_omni_v3_create_request_all_of import DockerWorkerConfigOmniV3CreateRequestAllOf
from lightly.openapi_generated.swagger_client.models.docker_worker_config_omni_v4_create_request import DockerWorkerConfigOmniV4CreateRequest
from lightly.openapi_generated.swagger_client.models.docker_worker_config_omni_v4_create_request_all_of import DockerWorkerConfigOmniV4CreateRequestAllOf
from lightly.openapi_generated.swagger_client.models.docker_worker_config_omni_vx_create_request import DockerWorkerConfigOmniVXCreateRequest
from lightly.openapi_generated.swagger_client.models.docker_worker_config_omni_vx_create_request_base import DockerWorkerConfigOmniVXCreateRequestBase
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v0_data import DockerWorkerConfigV0Data
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v0_data_all_of import DockerWorkerConfigV0DataAllOf
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v2 import DockerWorkerConfigV2
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v2_create_request import DockerWorkerConfigV2CreateRequest
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v2_data import DockerWorkerConfigV2Data
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v2_data_all_of import DockerWorkerConfigV2DataAllOf
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v2_docker import DockerWorkerConfigV2Docker
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v2_docker_datasource import DockerWorkerConfigV2DockerDatasource
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v2_docker_object_level import DockerWorkerConfigV2DockerObjectLevel
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v2_docker_stopping_condition import DockerWorkerConfigV2DockerStoppingCondition
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v2_lightly import DockerWorkerConfigV2Lightly
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v2_lightly_collate import DockerWorkerConfigV2LightlyCollate
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v2_lightly_model import DockerWorkerConfigV2LightlyModel
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v2_lightly_trainer import DockerWorkerConfigV2LightlyTrainer
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3 import DockerWorkerConfigV3
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_create_request import DockerWorkerConfigV3CreateRequest
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_data import DockerWorkerConfigV3Data
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_data_all_of import DockerWorkerConfigV3DataAllOf
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_datasource_input_expiration import DockerWorkerConfigV3DatasourceInputExpiration
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_docker import DockerWorkerConfigV3Docker
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_docker_corruptness_check import DockerWorkerConfigV3DockerCorruptnessCheck
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_docker_datasource import DockerWorkerConfigV3DockerDatasource
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_docker_training import DockerWorkerConfigV3DockerTraining
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_lightly import DockerWorkerConfigV3Lightly
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_lightly_checkpoint_callback import DockerWorkerConfigV3LightlyCheckpointCallback
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_lightly_collate import DockerWorkerConfigV3LightlyCollate
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_lightly_criterion import DockerWorkerConfigV3LightlyCriterion
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_lightly_loader import DockerWorkerConfigV3LightlyLoader
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_lightly_model import DockerWorkerConfigV3LightlyModel
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_lightly_optimizer import DockerWorkerConfigV3LightlyOptimizer
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v3_lightly_trainer import DockerWorkerConfigV3LightlyTrainer
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v4 import DockerWorkerConfigV4
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v4_data import DockerWorkerConfigV4Data
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v4_data_all_of import DockerWorkerConfigV4DataAllOf
from lightly.openapi_generated.swagger_client.models.docker_worker_config_v4_docker import DockerWorkerConfigV4Docker
from lightly.openapi_generated.swagger_client.models.docker_worker_config_vx_data import DockerWorkerConfigVXData
from lightly.openapi_generated.swagger_client.models.docker_worker_config_vx_data_base import DockerWorkerConfigVXDataBase
from lightly.openapi_generated.swagger_client.models.docker_worker_registry_entry_data import DockerWorkerRegistryEntryData
from lightly.openapi_generated.swagger_client.models.docker_worker_state import DockerWorkerState
from lightly.openapi_generated.swagger_client.models.docker_worker_type import DockerWorkerType
from lightly.openapi_generated.swagger_client.models.embedding2d_create_request import Embedding2dCreateRequest
from lightly.openapi_generated.swagger_client.models.embedding2d_data import Embedding2dData
from lightly.openapi_generated.swagger_client.models.embedding_data import EmbeddingData
from lightly.openapi_generated.swagger_client.models.expiry_handling_strategy_v3 import ExpiryHandlingStrategyV3
from lightly.openapi_generated.swagger_client.models.file_name_format import FileNameFormat
from lightly.openapi_generated.swagger_client.models.file_output_format import FileOutputFormat
from lightly.openapi_generated.swagger_client.models.filename_and_read_url import FilenameAndReadUrl
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
from lightly.openapi_generated.swagger_client.models.label_box_v4_data_row import LabelBoxV4DataRow
from lightly.openapi_generated.swagger_client.models.label_studio_task import LabelStudioTask
from lightly.openapi_generated.swagger_client.models.label_studio_task_data import LabelStudioTaskData
from lightly.openapi_generated.swagger_client.models.lightly_docker_selection_method import LightlyDockerSelectionMethod
from lightly.openapi_generated.swagger_client.models.lightly_model_v2 import LightlyModelV2
from lightly.openapi_generated.swagger_client.models.lightly_model_v3 import LightlyModelV3
from lightly.openapi_generated.swagger_client.models.lightly_trainer_precision_v2 import LightlyTrainerPrecisionV2
from lightly.openapi_generated.swagger_client.models.lightly_trainer_precision_v3 import LightlyTrainerPrecisionV3
from lightly.openapi_generated.swagger_client.models.prediction_singleton import PredictionSingleton
from lightly.openapi_generated.swagger_client.models.prediction_singleton_base import PredictionSingletonBase
from lightly.openapi_generated.swagger_client.models.prediction_singleton_classification import PredictionSingletonClassification
from lightly.openapi_generated.swagger_client.models.prediction_singleton_classification_all_of import PredictionSingletonClassificationAllOf
from lightly.openapi_generated.swagger_client.models.prediction_singleton_instance_segmentation import PredictionSingletonInstanceSegmentation
from lightly.openapi_generated.swagger_client.models.prediction_singleton_instance_segmentation_all_of import PredictionSingletonInstanceSegmentationAllOf
from lightly.openapi_generated.swagger_client.models.prediction_singleton_keypoint_detection import PredictionSingletonKeypointDetection
from lightly.openapi_generated.swagger_client.models.prediction_singleton_keypoint_detection_all_of import PredictionSingletonKeypointDetectionAllOf
from lightly.openapi_generated.swagger_client.models.prediction_singleton_object_detection import PredictionSingletonObjectDetection
from lightly.openapi_generated.swagger_client.models.prediction_singleton_object_detection_all_of import PredictionSingletonObjectDetectionAllOf
from lightly.openapi_generated.swagger_client.models.prediction_singleton_semantic_segmentation import PredictionSingletonSemanticSegmentation
from lightly.openapi_generated.swagger_client.models.prediction_singleton_semantic_segmentation_all_of import PredictionSingletonSemanticSegmentationAllOf
from lightly.openapi_generated.swagger_client.models.prediction_task_schema import PredictionTaskSchema
from lightly.openapi_generated.swagger_client.models.prediction_task_schema_base import PredictionTaskSchemaBase
from lightly.openapi_generated.swagger_client.models.prediction_task_schema_category import PredictionTaskSchemaCategory
from lightly.openapi_generated.swagger_client.models.prediction_task_schema_category_keypoints import PredictionTaskSchemaCategoryKeypoints
from lightly.openapi_generated.swagger_client.models.prediction_task_schema_category_keypoints_all_of import PredictionTaskSchemaCategoryKeypointsAllOf
from lightly.openapi_generated.swagger_client.models.prediction_task_schema_keypoint import PredictionTaskSchemaKeypoint
from lightly.openapi_generated.swagger_client.models.prediction_task_schema_keypoint_all_of import PredictionTaskSchemaKeypointAllOf
from lightly.openapi_generated.swagger_client.models.prediction_task_schema_simple import PredictionTaskSchemaSimple
from lightly.openapi_generated.swagger_client.models.prediction_task_schema_simple_all_of import PredictionTaskSchemaSimpleAllOf
from lightly.openapi_generated.swagger_client.models.prediction_task_schemas import PredictionTaskSchemas
from lightly.openapi_generated.swagger_client.models.questionnaire_data import QuestionnaireData
from lightly.openapi_generated.swagger_client.models.s3_region import S3Region
from lightly.openapi_generated.swagger_client.models.sama_task import SamaTask
from lightly.openapi_generated.swagger_client.models.sama_task_data import SamaTaskData
from lightly.openapi_generated.swagger_client.models.sample_create_request import SampleCreateRequest
from lightly.openapi_generated.swagger_client.models.sample_data import SampleData
from lightly.openapi_generated.swagger_client.models.sample_data_modes import SampleDataModes
from lightly.openapi_generated.swagger_client.models.sample_meta_data import SampleMetaData
from lightly.openapi_generated.swagger_client.models.sample_partial_mode import SamplePartialMode
from lightly.openapi_generated.swagger_client.models.sample_sort_by import SampleSortBy
from lightly.openapi_generated.swagger_client.models.sample_type import SampleType
from lightly.openapi_generated.swagger_client.models.sample_update_request import SampleUpdateRequest
from lightly.openapi_generated.swagger_client.models.sample_write_urls import SampleWriteUrls
from lightly.openapi_generated.swagger_client.models.sampling_config import SamplingConfig
from lightly.openapi_generated.swagger_client.models.sampling_config_stopping_condition import SamplingConfigStoppingCondition
from lightly.openapi_generated.swagger_client.models.sampling_create_request import SamplingCreateRequest
from lightly.openapi_generated.swagger_client.models.sampling_method import SamplingMethod
from lightly.openapi_generated.swagger_client.models.sector import Sector
from lightly.openapi_generated.swagger_client.models.selection_config import SelectionConfig
from lightly.openapi_generated.swagger_client.models.selection_config_all_of import SelectionConfigAllOf
from lightly.openapi_generated.swagger_client.models.selection_config_base import SelectionConfigBase
from lightly.openapi_generated.swagger_client.models.selection_config_entry import SelectionConfigEntry
from lightly.openapi_generated.swagger_client.models.selection_config_entry_input import SelectionConfigEntryInput
from lightly.openapi_generated.swagger_client.models.selection_config_entry_strategy import SelectionConfigEntryStrategy
from lightly.openapi_generated.swagger_client.models.selection_config_v3 import SelectionConfigV3
from lightly.openapi_generated.swagger_client.models.selection_config_v3_all_of import SelectionConfigV3AllOf
from lightly.openapi_generated.swagger_client.models.selection_config_v3_entry import SelectionConfigV3Entry
from lightly.openapi_generated.swagger_client.models.selection_config_v3_entry_input import SelectionConfigV3EntryInput
from lightly.openapi_generated.swagger_client.models.selection_config_v3_entry_strategy import SelectionConfigV3EntryStrategy
from lightly.openapi_generated.swagger_client.models.selection_config_v3_entry_strategy_all_of import SelectionConfigV3EntryStrategyAllOf
from lightly.openapi_generated.swagger_client.models.selection_config_v3_entry_strategy_all_of_target_range import SelectionConfigV3EntryStrategyAllOfTargetRange
from lightly.openapi_generated.swagger_client.models.selection_config_v4 import SelectionConfigV4
from lightly.openapi_generated.swagger_client.models.selection_config_v4_all_of import SelectionConfigV4AllOf
from lightly.openapi_generated.swagger_client.models.selection_config_v4_entry import SelectionConfigV4Entry
from lightly.openapi_generated.swagger_client.models.selection_config_v4_entry_input import SelectionConfigV4EntryInput
from lightly.openapi_generated.swagger_client.models.selection_config_v4_entry_strategy import SelectionConfigV4EntryStrategy
from lightly.openapi_generated.swagger_client.models.selection_input_predictions_name import SelectionInputPredictionsName
from lightly.openapi_generated.swagger_client.models.selection_input_type import SelectionInputType
from lightly.openapi_generated.swagger_client.models.selection_strategy_threshold_operation import SelectionStrategyThresholdOperation
from lightly.openapi_generated.swagger_client.models.selection_strategy_type import SelectionStrategyType
from lightly.openapi_generated.swagger_client.models.selection_strategy_type_v3 import SelectionStrategyTypeV3
from lightly.openapi_generated.swagger_client.models.service_account_basic_data import ServiceAccountBasicData
from lightly.openapi_generated.swagger_client.models.set_embeddings_is_processed_flag_by_id_body_request import SetEmbeddingsIsProcessedFlagByIdBodyRequest
from lightly.openapi_generated.swagger_client.models.shared_access_config_create_request import SharedAccessConfigCreateRequest
from lightly.openapi_generated.swagger_client.models.shared_access_config_data import SharedAccessConfigData
from lightly.openapi_generated.swagger_client.models.shared_access_type import SharedAccessType
from lightly.openapi_generated.swagger_client.models.tag_active_learning_scores_data import TagActiveLearningScoresData
from lightly.openapi_generated.swagger_client.models.tag_arithmetics_operation import TagArithmeticsOperation
from lightly.openapi_generated.swagger_client.models.tag_arithmetics_request import TagArithmeticsRequest
from lightly.openapi_generated.swagger_client.models.tag_arithmetics_response import TagArithmeticsResponse
from lightly.openapi_generated.swagger_client.models.tag_bit_mask_response import TagBitMaskResponse
from lightly.openapi_generated.swagger_client.models.tag_change_data import TagChangeData
from lightly.openapi_generated.swagger_client.models.tag_change_data_arithmetics import TagChangeDataArithmetics
from lightly.openapi_generated.swagger_client.models.tag_change_data_initial import TagChangeDataInitial
from lightly.openapi_generated.swagger_client.models.tag_change_data_metadata import TagChangeDataMetadata
from lightly.openapi_generated.swagger_client.models.tag_change_data_operation_method import TagChangeDataOperationMethod
from lightly.openapi_generated.swagger_client.models.tag_change_data_rename import TagChangeDataRename
from lightly.openapi_generated.swagger_client.models.tag_change_data_sampler import TagChangeDataSampler
from lightly.openapi_generated.swagger_client.models.tag_change_data_samples import TagChangeDataSamples
from lightly.openapi_generated.swagger_client.models.tag_change_data_scatterplot import TagChangeDataScatterplot
from lightly.openapi_generated.swagger_client.models.tag_change_data_upsize import TagChangeDataUpsize
from lightly.openapi_generated.swagger_client.models.tag_change_entry import TagChangeEntry
from lightly.openapi_generated.swagger_client.models.tag_create_request import TagCreateRequest
from lightly.openapi_generated.swagger_client.models.tag_creator import TagCreator
from lightly.openapi_generated.swagger_client.models.tag_data import TagData
from lightly.openapi_generated.swagger_client.models.tag_update_request import TagUpdateRequest
from lightly.openapi_generated.swagger_client.models.tag_upsize_request import TagUpsizeRequest
from lightly.openapi_generated.swagger_client.models.task_type import TaskType
from lightly.openapi_generated.swagger_client.models.team_basic_data import TeamBasicData
from lightly.openapi_generated.swagger_client.models.team_data import TeamData
from lightly.openapi_generated.swagger_client.models.team_role import TeamRole
from lightly.openapi_generated.swagger_client.models.trigger2d_embedding_job_request import Trigger2dEmbeddingJobRequest
from lightly.openapi_generated.swagger_client.models.update_docker_worker_registry_entry_request import UpdateDockerWorkerRegistryEntryRequest
from lightly.openapi_generated.swagger_client.models.update_team_membership_request import UpdateTeamMembershipRequest
from lightly.openapi_generated.swagger_client.models.user_type import UserType
from lightly.openapi_generated.swagger_client.models.video_frame_data import VideoFrameData
from lightly.openapi_generated.swagger_client.models.write_csv_url_data import WriteCSVUrlData
