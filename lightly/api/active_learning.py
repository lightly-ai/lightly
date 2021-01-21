""" Upload to Lightly API """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import *
import requests

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.api.utils import getenv


def upload_scores_to_api(scores: Dict[str, Iterable[float]]):
    raise NotImplementedError


def sampling_request_to_server(
        token: str,
        datasetId: str,
        tagId: str,
        embeddingId: str,
) -> str:
    """ Makes the sampling request to the server.

    Args:
        sampler_config (SamplerConfig):
            Contains info about batch size
        dataset_id (str):
            Unique identifier for the dataset.
        tag_id (str):
            Unique identifier for the current state (labelled_indices).
        embeddding_id (str):
            Unique identifier for previosuly uploaded embeddings.

    Returns:
        jobId if the request was successful.

    Raises:
        Runtime error response status.
    """
    server_location = getenv(
        'LIGHTLY_SERVER_LOCATION',
        'https://api.lightly.ai'
    )

    dst_url = (server_location +
               f"/v1/datasets/{datasetId}/tags/{tagId}/embeddings/{embeddingId}/sampling")

    SamplingCreateRequest = {
        'name': 'test',
        'method': 'RANDOM',
        'config': {
            'stoppingCondition': {
                'nSamples': 10,
                'minDistance': 5
            }
        },
    }

    response = requests.get(dst_url, params=SamplingCreateRequest)
    response_json = response.json()

    if response.status_code == 200:
        return response_json['jobId']
    raise RuntimeError(response.status_code)


from lightly.openapi_generated.swagger_client.configuration import Configuration
from lightly.openapi_generated.swagger_client.api_client import ApiClient
from lightly.openapi_generated.swagger_client.api.samplings_api import SamplingsApi
from lightly.openapi_generated.swagger_client.models import \
    sampling_config, sampling_method, sampling_create_request, sampling_config_stopping_condition


def sampling_request_to_server_with_openapi(
        token: str,
        dataset_id: str,
        tag_id: str,
        embedding_id: str,
) -> str:

    sampling_config_stopping_condition_ = sampling_config_stopping_condition.SamplingConfigStoppingCondition(
        n_samples=10, min_distance=5)
    sampling_config_ = sampling_config.SamplingConfig(stopping_condition=sampling_config_stopping_condition_)
    sampling_method_ = sampling_method.SamplingMethod.RANDOM
    sampling_create_request_ = sampling_create_request.SamplingCreateRequest(name='test', method=sampling_method_,
                                                                             config=sampling_config_)

    samplings_api = SamplingsApi()

    payload = sampling_create_request_
    payload.token = token
    response = samplings_api.trigger_sampling_by_id(payload, dataset_id, tag_id, embedding_id)

    response_json = response.json()

    if response.status_code == 200:
        return response_json['jobId']
    raise RuntimeError(response.status_code)


if __name__ == '__main__':
    token = 'f9b60358d529bdd824e3c2df'
    datasetId = '5ff6fa9b6580b3000acca8a8'
    tagId = 'initial-tag'
    embeddingId = '0'
    # print(sampling_request_to_server(token, datasetId, tagId, embeddingId))
    print(sampling_request_to_server_with_openapi(token, datasetId, tagId, embeddingId))
