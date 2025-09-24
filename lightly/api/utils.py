""" Communication Utility """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import io
import os
import random
import threading
import time
from enum import Enum
from typing import Iterator, List, Optional

# the following two lines are needed because
# PIL misidentifies certain jpeg images as MPOs
from PIL import JpegImagePlugin

JpegImagePlugin._getmp = lambda: None

from lightly.openapi_generated.swagger_client.configuration import Configuration

MAXIMUM_FILENAME_LENGTH = 255
RETRY_MAX_BACKOFF = 32
RETRY_MAX_RETRIES = 5


def retry(func, *args, **kwargs):  # type: ignore
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
    backoff = 1.0 + random.random() * 0.1
    max_backoff = RETRY_MAX_BACKOFF
    max_retries = RETRY_MAX_RETRIES

    # try to make the request
    current_retries = 0
    while True:
        try:
            # return on success
            return func(*args, **kwargs)
        except Exception as e:
            # sleep on failure
            time.sleep(backoff)
            backoff = 2 * backoff if backoff < max_backoff else backoff
            current_retries += 1

            # max retries exceeded
            if current_retries >= max_retries:
                raise RuntimeError(
                    f"Maximum retries exceeded! Original exception: {type(e)}: {str(e)}"
                ) from e


class Paginated(Iterator):
    def __init__(self, fn, page_size, *args, **kwargs):
        self.entries: List = []
        self.last_chunk_size = page_size
        self.offset = 0
        self.fn = fn
        self.page_size = page_size
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.entries) == 0:
            # stop iteration if the last chunk was smaller than the page size
            if self.last_chunk_size < self.page_size:
                raise StopIteration
            chunk = retry(
                self.fn,
                page_offset=self.offset * self.page_size,
                page_size=self.page_size,
                *self.args,
                **self.kwargs,
            )
            if len(chunk) == 0:
                raise StopIteration
            self.offset += 1
            self.last_chunk_size = len(chunk)
            # Handle the case where the chunk is a string. In this case we want
            # to return the whole page as a single string instead of an interable
            # of characters.
            chunk = chunk if not isinstance(chunk, str) else [chunk]
            self.entries.extend(chunk)
        return self.entries.pop(0)


def paginate_endpoint(fn, page_size=5000, *args, **kwargs) -> Iterator:
    """Paginates an API endpoint

    Args:
        fn:
            The endpoint which will be paginated until there is not any more data
        page_size:
            The size of the pages to pull
    """
    return Paginated(fn, page_size, *args, **kwargs)


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


def PIL_to_bytes(img, ext: str = "png", quality: int = None):
    """Return the PIL image as byte stream. Useful to send image via requests."""
    bytes_io = io.BytesIO()
    if quality is not None:
        img.save(bytes_io, format=ext, quality=quality)
    else:
        subsampling = -1 if ext.lower() in ["jpg", "jpeg"] else 0
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


def build_azure_signed_url_write_headers(
    content_length: str,
    x_ms_blob_type: str = "BlockBlob",
    accept: str = "*/*",
    accept_encoding: str = "*",
):
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
        "x-ms-blob-type": x_ms_blob_type,
        "Accept": accept,
        "Content-Length": content_length,
        "x-ms-original-content-length": content_length,
        "Accept-Encoding": accept_encoding,
    }
    return headers


class DatasourceType(Enum):
    S3 = "S3"
    GCS = "GCS"
    AZURE = "AZURE"
    LOCAL = "LOCAL"


def get_signed_url_destination(signed_url: str = "") -> DatasourceType:
    """
    Tries to figure out the of which cloud provider/datasource type a signed url comes from (S3, GCS, Azure)
    Args:
        signed_url:
            The signed url of a "bucket" provider
    Returns:
        DatasourceType
    """

    assert isinstance(signed_url, str)

    if "storage.googleapis.com/" in signed_url:
        return DatasourceType.GCS
    if ".amazonaws.com/" in signed_url and ".s3." in signed_url:
        return DatasourceType.S3
    if ".windows.net/" in signed_url:
        return DatasourceType.AZURE
    # default to local as it must be some special setup
    return DatasourceType.LOCAL


def get_lightly_server_location_from_env() -> str:
    return (
        getenv("LIGHTLY_SERVER_LOCATION", "https://api.lightly.ai").strip().rstrip("/")
    )


def get_api_client_configuration(
    token: Optional[str] = None,
    raise_if_no_token_specified: bool = True,
) -> Configuration:
    host = get_lightly_server_location_from_env()
    ssl_ca_cert = getenv("LIGHTLY_CA_CERTS", None)
    proxy = getenv("ALL_PROXY", getenv("HTTPS_PROXY", getenv("HTTP_PROXY", None)))

    if token is None:
        token = getenv("LIGHTLY_TOKEN", None)
    if token is None and raise_if_no_token_specified:
        raise ValueError(
            "Either provide a 'token' argument or export a LIGHTLY_TOKEN environment variable"
        )

    configuration = Configuration()
    configuration.api_key = {"ApiKeyAuth": token}
    configuration.ssl_ca_cert = ssl_ca_cert
    configuration.proxy = proxy
    configuration.host = host

    return configuration
