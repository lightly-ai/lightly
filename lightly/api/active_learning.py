""" Upload to Lightly API """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import *
import requests

from lightly.api.utils import getenv

from lightly.openapi_generated.swagger_client.configuration import Configuration
from lightly.openapi_generated.swagger_client.api_client import ApiClient
from lightly.openapi_generated.swagger_client.api.samplings_api import SamplingsApi

from lightly.openapi_generated.swagger_client.models.sampling_method import SamplingMethod
from lightly.openapi_generated.swagger_client.models.sampling_create_request import SamplingCreateRequest
from lightly.openapi_generated.swagger_client.models.sampling_config import SamplingConfig
from lightly.openapi_generated.swagger_client.models.sampling_config_stopping_condition import SamplingConfigStoppingCondition


def upload_scores_to_api(scores: Dict[str, Iterable[float]]):
    raise NotImplementedError


def sampling_request_to_server(token: str,
                               dataset_id: str,
                               tag_id: str,
                               embedding_id: str,
                               name: str,
                               method: str,
                               config: dict,
                               ) -> str:
    raise DeprecationWarning
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


def sampling_request_to_api(token: str,
                            dataset_id: str,
                            tag_id: str,
                            embedding_id: str,
                            sampling_request_name: str,
                            sampling_method: SamplingMethod = SamplingMethod.RANDOM,
                            stopping_condition_n_samples: int = None,
                            stopping_condition_min_distance: float = None
                            ) -> str:
    """ Makes the sampling request to the server using the generated openAPI

    Args:
        token (str):
            The token for authenticating the request.
        dataset_id (str):
            Unique identifier for the dataset.
        tag_id (str):
            Unique identifier for the current state (labelled_indices).
        embedding_id (str):
            Unique identifier for previosuly uploaded embeddings.
        sampling_request_name (str):
            Name describing the sampling request
        sampling_method (sampling_method.SamplingMethod):
            Choose from ['CORESET', 'RANDOM', 'BIT']
        stopping_condition_n_samples (int):
            the number of samples to sample
        stopping_condition_min_distance (float):
            the minimum distance chosen samples should have

    Returns:
        jobId if the request was successful.

    Raises:
        Runtime error response status.
    """

    sampling_config_stopping_condition_ = SamplingConfigStoppingCondition(
        n_samples=stopping_condition_n_samples, min_distance=stopping_condition_min_distance)
    sampling_config_ = SamplingConfig(stopping_condition=sampling_config_stopping_condition_)
    sampling_create_request_ = SamplingCreateRequest(
        name=sampling_request_name, method=sampling_method, config=sampling_config_)

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
    sampling_name = 'sampling-test'
    sampling_method = SamplingMethod.RANDOM
    n_samples = 100
    min_distance = 0.1
    response_job_id = sampling_request_to_api(token, datasetId, tagId, embeddingId,
                                              sampling_name, sampling_method, n_samples, min_distance)
    print(response_job_id)
