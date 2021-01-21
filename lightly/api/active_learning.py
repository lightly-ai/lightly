""" Upload to Lightly API """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import *
import requests

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.api.utils import getenv


def upload_scores_to_api(scores: Dict[str, Iterable[float]]):
    raise NotImplementedError


def sampling_request_to_server(token: str,
                               dataset_id: str,
                               tag_id: str,
                               embedding_id: str,
                               name: str,
                               method: str,
                               config: dict,
                               ) -> str :
    """ Makes the sampling request to the server.

    Args:
        token (str):
            The token for authenticating the request.
        dataset_id (str):
            Unique identifier for the dataset.
        tag_id (str):
            Unique identifier for the current state (labelled_indices).
        embeddding_id (str):
            Unique identifier for previosuly uploaded embeddings.
        name (str):
            Name describing the sampling request
        method (str):
            Choose from ['CORESET', 'RANDOM', 'BIT']
        config (dict):
            Contains info on stopping condition


    Returns:
        jobId if the request was successful.

    Raises:
        Runtime error response status.
    """
    server_location = getenv(
        'LIGHTLY_SERVER_LOCATION',
        'https://api-dev.lightly.ai'
    )
    server_location = 'https://app-dev.lightly.ai'

    dst_url = (server_location +
               f"/v1/datasets/{dataset_id}/tags/{tag_id}/embeddings/{embedding_id}/sampling")

    payload = {
        'name': name,
        'method': method,
        'config': config,
        'token': token
    }

    response = requests.post(dst_url, params=payload)

    if response.status_code == 200:
        response_json = response.json()
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

    configuration = Configuration()
    configuration.host = 'https://app-dev.lightly.ai'
    api_client = ApiClient(configuration=configuration, header_name='authorization', header_value=f"Bearer [{token}]")
    samplings_api = SamplingsApi(api_client=api_client)

    payload = sampling_create_request_
    response = samplings_api.trigger_sampling_by_id(payload, dataset_id, tag_id, embedding_id)

    return response.job_id


if __name__ == '__main__':
    token = 'bb10724138a5b33a0f35c444'
    datasetId = '6006f54aab0cd9000ad7914c'
    tagId = 'initial-tag'
    embeddingId = '0'
    name = 'sampling-test'
    method = 'RANDOM'
    config = {
            'stoppingCondition': {
                'nSamples': 10,
                'minDistance': 5,
            }
        }
    #print(sampling_request_to_server(token, datasetId, tagId,embeddingId, name, method,config))
    print(sampling_request_to_server_with_openapi(token, datasetId, tagId, embeddingId))
