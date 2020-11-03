""" Users Service """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import requests

from lightly.api.constants import LIGHTLY_MAXIMUM_DATASET_SIZE
from lightly.api.utils import getenv


def _prefix(*args, **kwargs):
    """Returns the prefix for the users routes.

    All routes through users require authentication via jwt.

    """
    server_location = getenv(
        'LIGHTLY_SERVER_LOCATION',
        'https://api.lightly.ai'
    )
    return server_location + '/users'


def get_quota(token: str):
    """Returns a dictionary with the quota for a user.

    The quota defines limitations for the current user.

    Args:
        token:
            A token to identify the user.

    Returns:
        A dictionary with the quota for the user.
    """
    dst_url = _prefix()
    payload = {
        'token': token
    }

    try:
        response = requests.get(dst_url, params=payload)
        return response.json()
    except Exception:
        return {'maxDatasetSize': LIGHTLY_MAXIMUM_DATASET_SIZE}
