""" Communication with Lightly Servers """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import time
import random
import requests

from lightly.api.utils import getenv
SERVER_LOCATION = getenv('LIGHTLY_SERVER_LOCATION',
                         'https://api.lightly.ai')


def _post_request(dst_url, data=None, json=None,
                  max_backoff=32, max_retries=5):

    counter = 0
    backoff = 1. + random.random() * 0.1
    success = False
    while not success:

        response = requests.post(dst_url, data=data, json=json)
        success = (response.status_code == 200)

        # exponential backoff
        if response.status_code in [500, 502]:
            time.sleep(backoff)
            backoff = 2*backoff if backoff < max_backoff else backoff
        elif response.status_code in [402]:
            msg = f'Dataset limit reached. Failed to upload samples. '
            msg += f'Contact your account manager to upgrade your subscription'
            raise ConnectionRefusedError(msg)
        # something went wrong
        elif not success:
            msg = f'Failed POST request to {dst_url} with status_code '
            msg += f'{response.status_code}.'
            raise RuntimeError(msg)

        counter += 1
        if counter >= max_retries:
            break

    if not success:
        msg = f'The connection to the server at {dst_url} timed out. '
        raise RuntimeError(msg)

    return response


def _put_request(dst_url, data=None, params=None, json=None,
                 max_backoff=32, max_retries=5):

    counter = 0
    backoff = 1. + random.random() * 0.1
    success = False
    while not success:

        response = requests.put(dst_url, data=data, json=json, params=params)
        success = (response.status_code == 200)

        # exponential backoff
        if response.status_code in [500, 502]:
            time.sleep(backoff)
            backoff = 2*backoff if backoff < max_backoff else backoff
        # something went wrong
        elif not success:
            msg = f'Failed PUT request to {dst_url} with status_code '
            msg += f'{response.status_code}.'
            raise RuntimeError(msg)

        counter += 1
        if counter >= max_retries:
            break

    if not success:
        msg = f'The connection to the server at {dst_url} timed out. '
        raise RuntimeError(msg)

    return response.status_code == 200


def _get_request(dst_url, params=None,
                 max_backoff=32, max_retries=5):

    counter = 0
    backoff = 1. + random.random() * 0.1
    success = False
    while not success:

        response = requests.get(dst_url, params=params)
        success = (response.status_code == 200)

        # exponential backoff
        if response.status_code in [500, 502]:
            time.sleep(backoff)
            backoff = 2*backoff if backoff < max_backoff else backoff
        # something went wrong
        elif not success:
            msg = f'Failed GET request to {dst_url} with status_code '
            msg += f'{response.status_code}.'
            raise RuntimeError(msg)

        counter += 1
        if counter >= max_retries:
            break

    if not success:
        msg = f'The connection to the server at {dst_url} timed out. '
        raise RuntimeError(msg)

    return response


def create_initial_tag(dataset_id: str, token: str):
    """Makes empty post request to dataset to create initial tag.

    Args:
        dataset_id:
            Identifier of the dataset.
        token:
            The token for authenticating the request.

    Returns:
        The response from the server.

    Raises:
        RuntimeError if creation of initial tag failed.

    """
    payload = {
        'token': token
    }
    dst_url = f'{SERVER_LOCATION}/users/datasets/{dataset_id}/tags'
    response = _post_request(dst_url, json=payload)
    return response


def get_presigned_upload_url(filename: str,
                             dataset_id: str,
                             sample_id: str,
                             token: str) -> str:
    """Creates and returns a signed url to upload an image to a dataset.

    Args:
        dataset_id:
            Identifier of the dataset.
        token:
            The token for authenticating the request.

    Returns:
        A string containing the signed url.

    Raises:
        RuntimeError if requesting signed url failed.
    """
    payload = {
        'fileName': filename,
        'token': token
    }

    dst_url = f'{SERVER_LOCATION}/users/'
    dst_url += f'datasets/{dataset_id}/'
    dst_url += f'samples/{sample_id}/writeurl'

    response = _get_request(dst_url, params=payload)
    signed_url = response.json()['signedWriteUrl']
    return signed_url


def upload_file_with_signed_url(file, url: str) -> bool:
    """Upload a file to the cloud storage using a signed URL.

    Args:
        filename:
            Path to a file for upload.
        url:
            Signed url for push.

    Returns:
        A boolean value indicating successful upload.

    Raises:
        RuntimeError if put request failed.
    """
    response = _put_request(url, data=file)
    file.close()
    return response


def upload_sample_with_metadata(filename: str,
                                thumbname: str,
                                metadata: str,
                                dataset_id: str,
                                token: str):
    """Upload a sample and its metadata to the servers.

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
    payload = {
        'sample': {
            'fileName': filename,
            'meta': metadata,
        },
        'token': token
    }
    if thumbname is not None:
        payload['sample']['thumbName'] = thumbname

    dst_url = f'{SERVER_LOCATION}/users/'
    dst_url += f'datasets/{dataset_id}/samples/'

    response = _post_request(dst_url, json=payload)
    sample_id = response.json()['sampleId']
    return sample_id


def upload_embedding(data: dict) -> bool:
    """Uploads a batch of embeddings to the servers.

    Args:
        data:
            Object with embedding data.

    Returns:
        A boolean value indicating successful upload.

    Raises:
        RuntimeError if upload was not successful.
    """
    payload = {
        'embeddingName': data['embeddingName'],
        'embeddings': data['embeddings'],
        'token': data['token'],
        'append': data['append'],
    }
    dataset_id = data['datasetId']
    dst_url = f'{SERVER_LOCATION}/users/datasets/{dataset_id}/embeddings'
    response = _post_request(dst_url, json=payload)
    return response


def put_image_type(dataset_id: str,
                   token: str,
                   img_type: str):
    """Add the attribute imgType to the db dataset entry.

    Args:
        dataset_id:
            Identifier of the dataset.
        token:
            The token for authenticating the request.
        img_type:
            Whether the sample was fully uploaded (full), only a thumbnail 
            (thumbnail) or only metadata (meta).

    Returns:
        A boolean value indicating a successful put request.

    Raises:
        RuntimeError if put was not successful.
    """
    params = {
        'token': token
    }
    data = {
        'dataset': {
            'imgType': img_type
        }
    }
    dst_url = f'{SERVER_LOCATION}/users/datasets/{dataset_id}'
    response = _put_request(dst_url, json=data, params=params)
    return response


def get_tags(dataset_id: str,
             token: str):
    """Returns all tags in a given dataset.

    Args:
        dataset_id:
            Identifier of the dataset.
        token:
            The token for authenticating the request.
    
    Returns:
        A list of tags for the dataset.
    
    Raises:
        RuntimeError if get request was not successful.
    """
    payload = {'token': token}
    dst_url = f'{SERVER_LOCATION}/users/datasets/{dataset_id}/tags/'
    response = _get_request(dst_url, params=payload)
    return response.json()


def get_samples(dataset_id: str,
                token: str,
                tag_name: str = 'initial-tag'):
    """Returns all samples in a dataset for a given tag.

    Args:
        dataset_id:
            Identifier of the dataset.
        token:
            The token for authenticating the request.
        tag_name:
            Name of the tag for which samples are requested.
    
    Returns:
        List of all sample names in a given tag.

    Raises:
        RuntimeError if get request was not successful, tag didn't exist, or
        there several tags with the same name.

    """

    # get tag_id
    tags = get_tags(dataset_id, token)
    tag_ids = [t['_id'] for t in tags if t['name'] == tag_name]
    if len(tag_ids) == 0:
        msg = f'No tag with name {tag_name} found '
        msg += f'for datset with id {dataset_id}'
        raise RuntimeError(msg)
    elif len(tag_ids) > 1:
        msg = f'{len(tag_ids)} tags with name {tag_name} found '
        msg += f'for dataset with id {dataset_id}'
        raise RuntimeError(msg)
    tag_id = tag_ids[0]

    # get files in tag
    payload = {'token': token}
    dst_url = f'{SERVER_LOCATION}/users/datasets/'
    dst_url += f'{dataset_id}/tags/{tag_id}/download'

    response = _get_request(dst_url, params=payload)
    return response.text.splitlines()


def get_latest_version(version, timeout: int = 1):
    """Returns the latest version of the lightly package.

    Args:
        timeout:
            Delay after which the request should timeout.
    
    Returns:
        The latest version if the request was successful otherwise None.
    
    """

    dst_url = f'{SERVER_LOCATION}/pip/version'
    payload = {'version': version}
    try:
        response = requests.get(dst_url, params=payload, timeout=timeout)
        return response.json()[0]
    except Exception:
        return None

def get_user_quota(token: str):
    """Returns a dictionary with the quota for a user.

    The quota defines limitations for the current user.
    
    Args:
        token:
            A token to identify the user.

    Returns:
        A dictionary with the quota for the user.
    """
    dst_url = f'{SERVER_LOCATION}/users/quota'
    payload = {'token': token}
    try:
        response = requests.get(dst_url, params=payload)
        return response.json()
    except Exception:
        return None


def get_embedding_summaries(dataset_id: str,
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
    dst_url = f'{SERVER_LOCATION}/users/datasets/'
    dst_url += f'{dataset_id}/embeddings/'
    payload = {
        'token': token,
        'mode': 'summaries'
    }
    response = _get_request(dst_url, params=payload)
    return response.json()