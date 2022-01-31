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

MAXIMUM_FILENAME_LENGTH = 255
RETRY_MAX_BACKOFF = 32
RETRY_MAX_RETRIES = 5

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
    max_backoff = RETRY_MAX_BACKOFF
    max_retries = RETRY_MAX_RETRIES

    # try to make the request
    for i in range(max_retries):
        try:
            # return on success
            return func(*args, **kwargs)
        except Exception:
            print('RETRYING!')
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



def build_azure_signed_url_write_headers(content_length: str,
                                         x_ms_blob_type: str = 'BlockBlob',
                                         accept: str = '*/*',
                                         accept_encoding: str = '*'):
    """Builds the headers required for a SAS PUT to Azure blob storage.

    Args:
        content_length:
            Length of the content in bytes as string.
        x_ms_blob_type:
            Blob type (one of BlockBlob, PageBlob, AppendBlob)
        accept:
            Indicates which content types the client is able to understand.
        accept_encoding:
            Indicates the content encoding that the client can understand.

    Returns:
        Formatted header which should be passed to the PUT request.

    """
    headers = {
        'x-ms-blob-type': x_ms_blob_type,
        'Accept': accept,
        'Content-Length': content_length,
        'x-ms-original-content-length': content_length,
        'Accept-Encoding': accept_encoding,
    }
    return headers