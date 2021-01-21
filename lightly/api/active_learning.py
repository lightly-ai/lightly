""" Upload to Lightly API """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import *
import requests

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.api.utils import create_api_client
from lightly.openapi_generated.swagger_client import SamplingMethod
from lightly.openapi_generated.swagger_client.api_client import ApiClient

from lightly.openapi_generated.swagger_client.api.samplings_api import SamplingsApi
from lightly.openapi_generated.swagger_client.models.sampling_create_request import SamplingCreateRequest

from lightly.openapi_generated.swagger_client.models.job_status_data import JobStatusData
from lightly.openapi_generated.swagger_client.api.jobs_api import JobsApi


def upload_scores_to_api(api_client: ApiClient, scores: Dict[str, Iterable[float]]):
    pass  # raise NotImplementedError


def get_job_status_from_api(api_client: ApiClient, job_id: str) -> JobStatusData:
    jobs_api = JobsApi(api_client)
    job_status_data = jobs_api.get_job_status_by_id(job_id=job_id)
    return job_status_data


def sampling_request_to_api(api_client: ApiClient,
                            dataset_id: str,
                            embedding_id: str,
                            sampler_config: SamplerConfig,
                            preselected_tag_id: str = None,
                            query_tag_id: str = None
                            ) -> str:
    """ Makes the sampling request to the server using the generated openAPI

    Args:
        api_client (str):
            The client for accessing the api.
        dataset_id (str):
            Unique identifier for the dataset.
        tag_id (str):
            Unique identifier for the current state (labelled_indices).
        embedding_id (str):
            Unique identifier for previosuly uploaded embeddings.

    Returns:
        jobId if the request was successful.

    Raises:
        Runtime error response status.
    """

    samplings_api = SamplingsApi(api_client=api_client)
    payload = sampler_config.get_as_api_sampling_create_request(
        preselected_tag_id=preselected_tag_id, query_tag_id=query_tag_id)
    response = samplings_api.trigger_sampling_by_id(payload, dataset_id, embedding_id)
    return response.job_id


if __name__ == '__main__':
    token = 'bb10724138a5b33a0f35c444'
    api_client = create_api_client(token, host='https://app-dev.lightly.ai')
    datasetId = '6006f54aab0cd9000ad7914c'
    tagId = 'initial-tag'
    embeddingId = '0'
    sampler_config = SamplerConfig(method=SamplingMethod.RANDOM, batch_size=32, min_distance=-1, name='sampling_test')
    response_job_id = sampling_request_to_api(api_client, datasetId, tagId, embeddingId, sampler_config)
    print(response_job_id)
