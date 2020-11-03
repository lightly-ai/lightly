""" Docker Service """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import requests
from lightly.api.utils import getenv

def _prefix(*args, **kwargs):
    """Returns the prefix for the users routes.

    All routes through users require authentication via jwt.

    """
    server_location = getenv(
        'LIGHTLY_SERVER_LOCATION',
        'https://api.lightly.ai'
    )
    return server_location + '/users/docker'


def get_soft_authorization(token: str):
    """Makes a call to the api to request authorization to run the container.

    Args:
        token:
            User access token.

    Returns:
        The server's response and the response status.

    """
    dst_url = _prefix() + '/soft_authorization'
    payload = {
        'token': token,
    }

    response = requests.get(dst_url, params=payload)
    status = response.status_code
    return response.json(), status


def get_authorization(token: str,
                      timestamp: str,
                      task_description: dict):
    """Makes a call to the api to request authorization to run the container.

    Args:
        token:
            User access token.
        timestamp:
            Access request timestamp.
        task_description:
            A dictionary describing the task.

    Returns:
        The body of the request, the server's response and the response status.

    """
    dst_url = _prefix() + '/authorization'
    payload = {
        'token': token,
    }
    json = {
        'token': token,
        'timestamp': timestamp,
        'task_description': task_description,
    }

    response = requests.get(dst_url, params=payload, json=json)
    status = response.status_code
    return json, response.json(), status
