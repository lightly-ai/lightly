""" Embeddings Service """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
from typing import *
import requests
from datetime import datetime

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
                             name: Union[str, None] = None) -> Tuple[int, str, str]:
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

    if name is None:
        date_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
        name = f"embedding_{date_time}"

    payload['name'] = name

    response = requests.get(dst_url, params=payload)
    response_json = response.json()
    status = response.status_code

    if status == 200:
        signed_url = response_json['signedWriteUrl']
        embedding_id = response_json['embeddingId']
    else:
        raise ValueError(f"getting the presigned url failed with status {status}")

    return status, signed_url, embedding_id
