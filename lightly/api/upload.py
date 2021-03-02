""" Upload to Lightly API """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
import copy
import time
import random
from typing import Union

import numpy as np
import torchvision

from itertools import islice

#import lightly.api.routes as routes
from lightly.api import routes
from lightly.api.constants import LIGHTLY_MAXIMUM_DATASET_SIZE

from lightly.api.utils import get_thumbnail_from_img
from lightly.api.utils import getenv
from lightly.api.utils import check_image
from lightly.api.utils import check_filename
from lightly.api.utils import PIL_to_bytes
from lightly.api.utils import put_request
from lightly.openapi_generated.swagger_client import SamplesApi, SampleCreateRequest
from lightly.openapi_generated.swagger_client.configuration import Configuration
from lightly.openapi_generated.swagger_client.models.initial_tag_create_request import InitialTagCreateRequest

from lightly.data import LightlyDataset

from lightly.openapi_generated.swagger_client.api_client import ApiClient
from lightly.openapi_generated.swagger_client.api.tags_api import TagsApi

import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor


def get_api_client(token: str, host: str = None) -> ApiClient:
    if host is None:
        host = getenv('LIGHTLY_SERVER_LOCATION', 'https://api.lightly.ai')
    configuration = Configuration()
    configuration.host = host
    configuration.api_key = {'token': token}
    api_client = ApiClient(configuration=configuration)
    return api_client


def _upload_single_image(image,
                         label,
                         filename,
                         dataset_id,
                         token,
                         mode):
    """Uploads a single image to the Lightly platform.

    """

    # check whether the filename is too long
    basename = filename
    if not check_filename(basename):
        msg = (f'Filename {basename} is longer than the allowed maximum of '
               'characters and will be skipped.')
        warnings.warn(msg)
        return False

    # calculate metadata, and check if corrupted
    metadata = check_image(image)

    # generate thumbnail if necessary
    thumbname = None
    thumbnail = None
    if mode == 'thumbnails' and not metadata['is_corrupted']:
        thumbname = '.'.join(basename.split('.')[:-1]) + '_thumb.webp'
        thumbnail = get_thumbnail_from_img(image)

    # upload sample with metadata
    sample_upload_success = True

    api_client = get_api_client(token=token)
    samples_api = SamplesApi(api_client=api_client)

    try:
        body = SampleCreateRequest(file_name=basename, thumb_name=thumbname, meta_data=metadata)
        sample_id = samples_api.create_sample_by_dataset_id(body=body, dataset_id=dataset_id).id
    except RuntimeError:
        raise ValueError("Creating the sampling in the web platform failed.")

    # upload thumbnail
    thumbnail_upload_success = True
    if mode == 'thumbnails' and not metadata['is_corrupted'] and sample_upload_success:
        try:
            # try to get signed url for thumbnail
            signed_url = samples_api.\
                get_sample_image_write_url_by_id(dataset_id=dataset_id, sample_id=sample_id, is_thumbnail=True)
            # try to upload thumbnail
            upload_file_with_signed_url(
                PIL_to_bytes(thumbnail, ext='webp', quality=70),
                signed_url
            )
            # close thumbnail
            thumbnail.close()
        except RuntimeError:
            thumbnail_upload_success = False

    # upload full image
    image_upload_success = True
    if mode == 'full' and not metadata['is_corrupted']:
        try:
            # try to get signed url for image
            signed_url = samples_api. \
                get_sample_image_write_url_by_id(dataset_id=dataset_id, sample_id=sample_id, is_thumbnail=False)

            # try to upload image
            upload_file_with_signed_url(
                PIL_to_bytes(image),
                signed_url
            )
            # close image
            image.close()
        except RuntimeError:
            image_upload_success = False

    success = sample_upload_success
    success = success and thumbnail_upload_success
    success = success and image_upload_success
    return success



def upload_file_with_signed_url(file, url: str) -> bool:
    """Upload a file to the cloud storage using a signed URL.

    Args:
        file:
            The buffered file reader to upload.
        url:
            Signed url for push.

    Returns:
        A boolean value indicating successful upload.

    Raises:
        RuntimeError if put request failed.
    """
    response = put_request(url, data=file)
    file.close()
    return response
