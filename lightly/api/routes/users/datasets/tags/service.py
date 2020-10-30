""" Tags Service """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from . import _prefix
import lightly.api.utils as utils


def get(dataset_id: str,
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
    dst_url = _prefix(dataset_id=dataset_id)
    payload = {
        'token': token
    }

    # fix url, TODO: fix api instead
    dst_url += '/'

    response = utils.get_request(dst_url, params=payload)
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
    tags = get(dataset_id, token)
    tag_ids = [t['_id'] for t in tags if t['name'] == tag_name]

    if len(tag_ids) == 0:
        msg = f'No tag with name {tag_name} found '
        msg += f'for datset with id {dataset_id}'
        raise RuntimeError(msg)

    if len(tag_ids) > 1:
        msg = f'{len(tag_ids)} tags with name {tag_name} found '
        msg += f'for dataset with id {dataset_id}'
        raise RuntimeError(msg)

    tag_id = tag_ids[0]

    # get files in tag
    dst_url = _prefix(dataset_id=dataset_id, tag_id=tag_id)
    dst_url += '/download'
    payload = {
        'token': token
    }

    response = utils.get_request(dst_url, params=payload)
    return response.text.splitlines()


def post(dataset_id: str,
         token: str,
         tag: dict = None):
    """Makes post request to dataset to create a tag.

    Args:
        dataset_id:
            Identifier of the dataset.
        token:
            The token for authenticating the request.
        tag:
            Description of the tag, dictionary with previous tag, name, list
            of samples ids, and changes. If None, the initial-tag will be
            created.

    Returns:
        The response from the server.

    Raises:
        RuntimeError if the request was not successful.

    """
    dst_url = _prefix(dataset_id=dataset_id)
    payload = {
        'token': token
    }

    if tag is not None:
        payload['tag'] = tag

    response = utils.post_request(dst_url, json=payload)
    return response
