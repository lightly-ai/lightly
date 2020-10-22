""" Embeddings Service """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from . import _prefix
import lightly.api.utils as utils


def get_summaries(dataset_id: str,
                  token: str):
    """Returns a list of all embedding summaries for a dataset.

    Args:
        dataset_id:
            Identifier of the dataset.
        token:
            The token for authenticating the request.

    Returns:
        A list of all embedding summaries for the requested dataset.

    Raises:
        RuntimeError if the get request was not successful.

    """
    dst_url = _prefix(dataset_id=dataset_id)
    payload = {
        'token': token,
        'mode': 'summaries'
    }

    # fix url, TODO: fix api instead
    dst_url += '/'

    response = utils.get_request(dst_url, params=payload)
    return response.json()


def post(dataset_id: str,
         token: str,
         data: dict) -> bool:
    """Uploads a batch of embeddings to the servers.

    Args:
        dataset_id:
            Identifier of the dataset.
        token:
            The token for authenticating the request.
        data:
            Object with embedding data.

    Returns:
        A boolean value indicating successful upload.

    Raises:
        RuntimeError if upload was not successful.
    """
    dst_url = _prefix(dataset_id=dataset_id)
    payload = {
        'embeddingName': data['embeddingName'],
        'embeddings': data['embeddings'],
        'append': data['append'],
        'token': token,
    }

    response = utils.post_request(dst_url, json=payload)
    return response
