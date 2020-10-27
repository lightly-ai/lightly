""" PIP Package Service """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from . import _prefix
import requests


def get_version(version, timeout: int = 1):
    """Returns the latest version number of the lightly package.

    Args:
        version:
            The local version of the lightly package.
        timeout:
            Delay after which the request should timeout.

    Returns:
        The latest version number as a string or None.

    """
    dst_url = _prefix() + '/version'
    payload = {'version': version}

    try:
        response = requests.get(dst_url, params=payload, timeout=timeout)
        return response.json()[0]
    except Exception:
        return None
