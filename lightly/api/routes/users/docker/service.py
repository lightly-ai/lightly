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

    response = requests.post(dst_url, params=payload, json=json)
    status = response.status_code
    return json, response.json(), status


def post_diagnostics(token: str,
                     run_id: str,
                     action: str,
                     data: dict,
                     timestamp: str):
    """Make a call to the api to add a diagnostics entry

    Args:
        token:
            User access token.
        run_id:
            Unique id of the docker run.
        action:
            Description of the diagnostics.
        data:
            Relevant extra data.
        timestamp:
            Timestamp of the diagnostics.

    Returns:
        True if the post was successful, false otherwise.

    """
    dst_url = _prefix()
    payload = {
        'token': token,
        'runId': run_id,
        'action': action,
        'data': data,
        'timestamp': timestamp,
    }

    try:
        response = requests.post(
            dst_url,
            json=payload,
        )
    except Exception:
        return False
    else:
        return (response.status_code == 200)