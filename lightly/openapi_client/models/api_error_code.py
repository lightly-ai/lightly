# coding: utf-8

"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""


from inspect import getfullargspec
import pprint
import re  # noqa: F401
from enum import Enum
from aenum import no_arg  # type: ignore





class ApiErrorCode(str, Enum):
    """
    ApiErrorCode
    """

    """
    allowed enum values
    """
    BAD_REQUEST = 'BAD_REQUEST'
    NOT_IMPLEMENTED = 'NOT_IMPLEMENTED'
    FORBIDDEN = 'FORBIDDEN'
    UNAUTHORIZED = 'UNAUTHORIZED'
    NOT_FOUND = 'NOT_FOUND'
    MALFORMED_REQUEST = 'MALFORMED_REQUEST'
    MALFORMED_RESPONSE = 'MALFORMED_RESPONSE'
    PAYLOAD_TOO_LARGE = 'PAYLOAD_TOO_LARGE'
    JWT_INVALID = 'JWT_INVALID'
    JWT_MALFORMED = 'JWT_MALFORMED'
    CREATION_FAILED = 'CREATION_FAILED'
    JOB_CREATION_FAILED = 'JOB_CREATION_FAILED'
    JOB_UNKNOWN = 'JOB_UNKNOWN'
    USER_NOT_KNOWN = 'USER_NOT_KNOWN'
    USER_ACCOUNT_DEACTIVATED = 'USER_ACCOUNT_DEACTIVATED'
    USER_ACCOUNT_BLOCKED = 'USER_ACCOUNT_BLOCKED'
    TEAM_ACCOUNT_PLAN_INSUFFICIENT = 'TEAM_ACCOUNT_PLAN_INSUFFICIENT'
    ILLEGAL_ACTION_RESOURCE_IN_USE = 'ILLEGAL_ACTION_RESOURCE_IN_USE'
    DATASET_UNKNOWN = 'DATASET_UNKNOWN'
    DATASET_NOT_SUPPORTED = 'DATASET_NOT_SUPPORTED'
    DATASET_TAG_INVALID = 'DATASET_TAG_INVALID'
    DATASET_NAME_EXISTS = 'DATASET_NAME_EXISTS'
    DATASET_AT_MAX_CAPACITY = 'DATASET_AT_MAX_CAPACITY'
    DATASET_DATASOURCE_UNKNOWN = 'DATASET_DATASOURCE_UNKNOWN'
    DATASET_DATASOURCE_CREDENTIALS_ERROR = 'DATASET_DATASOURCE_CREDENTIALS_ERROR'
    DATASET_DATASOURCE_INVALID = 'DATASET_DATASOURCE_INVALID'
    DATASET_DATASOURCE_ACTION_NOT_IMPLEMENTED = 'DATASET_DATASOURCE_ACTION_NOT_IMPLEMENTED'
    DATASET_DATASOURCE_ILLEGAL_ACTION = 'DATASET_DATASOURCE_ILLEGAL_ACTION'
    DATASET_DATASOURCE_RELEVANT_FILENAMES_INVALID = 'DATASET_DATASOURCE_RELEVANT_FILENAMES_INVALID'
    ACCESS_CONTROL_UNKNOWN = 'ACCESS_CONTROL_UNKNOWN'
    EMBEDDING_UNKNOWN = 'EMBEDDING_UNKNOWN'
    EMBEDDING_NAME_EXISTS = 'EMBEDDING_NAME_EXISTS'
    EMBEDDING_INVALID = 'EMBEDDING_INVALID'
    EMBEDDING_NOT_READY = 'EMBEDDING_NOT_READY'
    EMBEDDING_ROW_COUNT_UNKNOWN = 'EMBEDDING_ROW_COUNT_UNKNOWN'
    EMBEDDING_ROW_COUNT_INVALID = 'EMBEDDING_ROW_COUNT_INVALID'
    EMBEDDING_2_D_UNKNOWN = 'EMBEDDING_2D_UNKNOWN'
    TAG_UNKNOWN = 'TAG_UNKNOWN'
    TAG_NAME_EXISTS = 'TAG_NAME_EXISTS'
    TAG_INITIAL_EXISTS = 'TAG_INITIAL_EXISTS'
    TAG_UNDELETABLE_NOT_A_LEAF = 'TAG_UNDELETABLE_NOT_A_LEAF'
    TAG_UNDELETABLE_IS_INITIAL = 'TAG_UNDELETABLE_IS_INITIAL'
    TAG_NO_TAG_IN_DATASET = 'TAG_NO_TAG_IN_DATASET'
    TAG_PREVTAG_NOT_IN_DATASET = 'TAG_PREVTAG_NOT_IN_DATASET'
    TAG_QUERYTAG_NOT_IN_DATASET = 'TAG_QUERYTAG_NOT_IN_DATASET'
    TAG_PRESELECTEDTAG_NOT_IN_DATASET = 'TAG_PRESELECTEDTAG_NOT_IN_DATASET'
    TAG_NO_SCORES_AVAILABLE = 'TAG_NO_SCORES_AVAILABLE'
    SAMPLE_UNKNOWN = 'SAMPLE_UNKNOWN'
    SAMPLE_THUMBNAME_UNKNOWN = 'SAMPLE_THUMBNAME_UNKNOWN'
    SAMPLE_CREATE_REQUEST_INVALID_FORMAT = 'SAMPLE_CREATE_REQUEST_INVALID_FORMAT'
    SAMPLE_CREATE_REQUEST_INVALID_CROP_DATA = 'SAMPLE_CREATE_REQUEST_INVALID_CROP_DATA'
    PREDICTION_TASK_SCHEMA_UNKNOWN = 'PREDICTION_TASK_SCHEMA_UNKNOWN'
    PREDICTION_TASK_SCHEMA_CATEGORIES_NOT_UNIQUE = 'PREDICTION_TASK_SCHEMA_CATEGORIES_NOT_UNIQUE'
    SCORE_UNKNOWN = 'SCORE_UNKNOWN'
    DOCKER_RUN_UNKNOWN = 'DOCKER_RUN_UNKNOWN'
    DOCKER_RUN_DATASET_UNAVAILABLE = 'DOCKER_RUN_DATASET_UNAVAILABLE'
    DOCKER_RUN_REPORT_UNAVAILABLE = 'DOCKER_RUN_REPORT_UNAVAILABLE'
    DOCKER_RUN_ARTIFACT_UNKNOWN = 'DOCKER_RUN_ARTIFACT_UNKNOWN'
    DOCKER_RUN_ARTIFACT_EXISTS = 'DOCKER_RUN_ARTIFACT_EXISTS'
    DOCKER_RUN_ARTIFACT_UNAVAILABLE = 'DOCKER_RUN_ARTIFACT_UNAVAILABLE'
    DOCKER_WORKER_UNKNOWN = 'DOCKER_WORKER_UNKNOWN'
    DOCKER_WORKER_CONFIG_UNKNOWN = 'DOCKER_WORKER_CONFIG_UNKNOWN'
    DOCKER_WORKER_CONFIG_NOT_COMPATIBLE_WITH_DATASOURCE = 'DOCKER_WORKER_CONFIG_NOT_COMPATIBLE_WITH_DATASOURCE'
    DOCKER_WORKER_CONFIG_REFERENCES_INVALID_FILES = 'DOCKER_WORKER_CONFIG_REFERENCES_INVALID_FILES'
    DOCKER_WORKER_CONFIG_IN_USE = 'DOCKER_WORKER_CONFIG_IN_USE'
    DOCKER_WORKER_CONFIG_INVALID = 'DOCKER_WORKER_CONFIG_INVALID'
    DOCKER_WORKER_SCHEDULE_UNKNOWN = 'DOCKER_WORKER_SCHEDULE_UNKNOWN'
    DOCKER_WORKER_SCHEDULE_UPDATE_FAILED = 'DOCKER_WORKER_SCHEDULE_UPDATE_FAILED'
    METADATA_CONFIGURATION_UNKNOWN = 'METADATA_CONFIGURATION_UNKNOWN'
    CUSTOM_METADATA_AT_MAX_SIZE = 'CUSTOM_METADATA_AT_MAX_SIZE'
    ACCOUNT_SUBSCRIPTION_INSUFFICIENT = 'ACCOUNT_SUBSCRIPTION_INSUFFICIENT'
    TEAM_UNKNOWN = 'TEAM_UNKNOWN'


