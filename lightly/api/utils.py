""" Communication Utility """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import io
import os
import time
import random

import numpy as np
from PIL import Image, ImageFilter
# the following two lines are needed because
# PIL misidentifies certain jpeg images as MPOs
from PIL import JpegImagePlugin

JpegImagePlugin._getmp = lambda: None

MAXIMUM_FILENAME_LENGTH = 80


def retry(func, *args, **kwargs):
    """Repeats a function until it completes successfully or fails too often.

    Args:
        func:
            The function call to repeat.
        args:
            The arguments which are passed to the function.
        kwargs:
            Key-word arguments which are passed to the function.

    Returns:
        What func returns.

    Exceptions:
        RuntimeError when number of retries has been exceeded.

    """

    # config
    backoff = 1. + random.random() * 0.1
    max_backoff = 32
    max_retries = 5

    # try to make the request
    for i in range(max_retries):
        try:
            # return on success
            return func(*args, **kwargs)
        except Exception:
            # sleep on failure
            time.sleep(backoff)
            backoff = 2 * backoff if backoff < max_backoff else backoff
        
    # max retries exceeded
    raise RuntimeError('The connection to the server timed out.')


def getenv(key: str, default: str):
    """Return the value of the environment variable key if it exists,
       or default if it doesn’t.

    """
    try:
        return os.getenvb(key.encode(), default.encode()).decode()
    except Exception:
        pass
    try:
        return os.getenv(key, default)
    except Exception:
        pass
    return default


def PIL_to_bytes(img, ext: str = 'png', quality: int = None):
    """Return the PIL image as byte stream. Useful to send image via requests.

    """
    bytes_io = io.BytesIO()
    if quality is not None:
        img.save(bytes_io, format=ext, quality=quality)
    else:
        subsampling = -1 if ext.lower() in ['jpg', 'jpeg'] else 0
        img.save(bytes_io, format=ext, quality=100, subsampling=subsampling)
    bytes_io.seek(0)
    return bytes_io


def check_filename(basename):
    """Checks the length of the filename.

    Args:
        basename:
            Basename of the file.

    """
    return len(basename) <= MAXIMUM_FILENAME_LENGTH
