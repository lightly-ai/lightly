""" Samples Service """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.api.utils import getenv, get_request, post_request
from typing import Union


def _prefix(dataset_id: Union[str, None] = None,
            sample_id: Union[str, None] = None,
            *args, **kwargs):
    """Returns the prefix for the samples routes.

    Args:
        dataset_id:
            Identifier of the dataset.
        sample_id:
            Identifier of the sample.

    """
    server_location = getenv(
        'LIGHTLY_SERVER_LOCATION',
        'https://api.lightly.ai'
    )
    prefix = server_location + '/users/datasets'
    if dataset_id is None:
        prefix = prefix + '/samples'
    else:
        prefix = prefix + '/' + dataset_id + '/samples'
    if sample_id is None:
        return prefix
    else:
        return prefix + '/' + sample_id


def get_presigned_upload_url(filename: str,
                             dataset_id: str,
                             sample_id: str,
                             token: str) -> str:
    """Creates and returns a signed url to upload an image to a dataset.

    Args:
        filename:
            Filename of the image to upload.
        dataset_id:
            Identifier of the dataset.
        sample_id:
            Identifier of the sample.
        token:
            The token for authenticating the request.

    Returns:
        A string containing the signed url.

    Raises:
        RuntimeError if requesting signed url failed.
    """
    dst_url = _prefix(dataset_id=dataset_id, sample_id=sample_id) + '/writeurl'
    payload = {
        'fileName': filename,
        'token': token
    }

    response = get_request(dst_url, params=payload)
    signed_url = response.json()['signedWriteUrl']
    return signed_url


def post(filename: str,
         thumbname: str,
         metadata: str,
         dataset_id: str,
         token: str):
    """Uploads a sample and its metadata to the servers.

    Args:
        filename:
            Filename of the sample.
        thumbname:
            Filename of thumbnail if it exists.
        metadata:
            Dictionary containing metadata of the sample.
        dataset_id:
            Identifier of the dataset.
        token:
            The token for authenticating the request.

    Returns:
        Sample id of the uploaded sample.

    Raises:
        RuntimeError if post request failed.

    """
    dst_url = _prefix(dataset_id=dataset_id)
    payload = {
        'sample': {
            'fileName': filename,
            'meta': metadata,
        },
        'token': token
    }

    # fix url, TODO: fix api instead
    dst_url += '/'

    if thumbname is not None:
        payload['sample']['thumbName'] = thumbname

    response = post_request(dst_url, json=payload)
    sample_id = response.json()['sampleId']
    return sample_id
