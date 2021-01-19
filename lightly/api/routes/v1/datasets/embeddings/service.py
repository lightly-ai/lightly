""" Embeddings Service """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
import requests
from lightly.api.utils import getenv
from typing import Union


def _prefix(dataset_id: Union[str, None] = None,
            *args, **kwargs):
    """Returns the prefix for the embeddings routes.

    """
    server_location = getenv(
        'LIGHTLY_SERVER_LOCATION',
        'https://api.lightly.ai'
    )
    prefix = server_location + '/v1/datasets'
    if dataset_id is not None:
        prefix = prefix + '/' + dataset_id
    
    prefix = prefix + '/embeddings'
    return prefix


def get_presigned_upload_url(dataset_id: str,
                             token: str,
                             name: Union[str, None] = None):
    """Creates and returns a signed url to upload a csv file of embeddings.

    Args:
        dataset_id:
            Identifier of the dataset.
        token:
            The token for authenticating the request.
        name:
            The name of the embedding (None will resolve to "default").

    Returns:
        A signed url.

    """
    dst_url = _prefix(dataset_id=dataset_id) + '/writeCSVUrl'
    payload = {
        'datasetId': dataset_id,
        'token': token,
    }

    if name is not None:
        payload['name'] = name

    response = requests.get(dst_url, params=payload)
    response_json = response.json()

    if response.status_code == 200:
        signed_url = response_json['signedWriteUrl']
        # TODO use embeddingId?
    else:
        signed_url = None

    return signed_url, response.status_code

