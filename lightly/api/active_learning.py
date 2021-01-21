""" Upload to Lightly API """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import *
import requests

from lightly.active_learning.config.sampler_config import SamplerConfig
from lightly.api.utils import getenv


def upload_scores_to_api(scores: Dict[str, Iterable[float]]):
    raise NotImplementedError


def sampling_request_to_server(#sampler_create_request: SamplerCreateRequest,
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

if __name__ == '__main__':
    datasetId = '6006f54aab0cd9000ad7914c'
    tagId = 'initial-tag'
    embeddingId = '0'
    print(sampling_request_to_server(datasetId, tagId, embeddingId))
