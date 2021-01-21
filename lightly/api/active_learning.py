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

    dst_url = (server_location +
               f"/v1/datasets/{dataset_id}/tags/{tag_id}/embeddings/{embedding_id}/sampling")

    payload = {
        'name': name,
        'method': method,
        'config': config,
    }

    response = requests.post(dst_url, params=payload)
    response_json = response.json()

    if response.status_code == 200:
        return response_json['jobId']
    
    raise RuntimeError(response.status_code)

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
    print(sampling_request_to_server(token, datasetId, tagId,
                                     embeddingId, name, method,
                                     config))
