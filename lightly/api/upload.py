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

from lightly.utils import fit_pca
from lightly.utils import load_embeddings_as_dict

from lightly.data import LightlyDataset

from lightly.openapi_generated.swagger_client.api_client import ApiClient
from lightly.openapi_generated.swagger_client.api.tags_api import TagsApi

import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor

import PIL.Image as Image


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
        sample_id = samples_api.create_sample_by_dataset_id(body=body, dataset_id=dataset_id)
    except RuntimeError as e:
        sample_upload_success = False
        raise ValueError

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


def upload_dataset(dataset: LightlyDataset,
                   dataset_id: str,
                   token: str,
                   max_workers: int = 8,
                   mode: str = 'thumbnails',
                   verbose: bool = True):
    """Uploads images from a directory to the Lightly cloud solution.

    Args:
        dataset
            The dataset to upload.
        dataset_id:
            The unique identifier for the dataset.
        token:
            Token for authentication.
        max_workers:
            Maximum number of workers uploading images in parallel.
        max_requests:
            Maximum number of requests a single worker can do before he has
            to wait for the others.
        mode:
            One of [full, thumbnails, metadata]. Whether to upload thumbnails,
            full images, or metadata only.

    Raises:
        ValueError if dataset is too large.
        RuntimeError if the connection to the server failed.
        RuntimeError if dataset already has an initial tag.

    """

    # check the allowed dataset size
    api_max_dataset_size, status_code = routes.users.get_quota(token)
    max_dataset_size = min(api_max_dataset_size, LIGHTLY_MAXIMUM_DATASET_SIZE)
    if len(dataset) > max_dataset_size:
        msg = f'Your dataset has {len(dataset)} samples which'
        msg += f' is more than the allowed maximum of {max_dataset_size}'
        raise ValueError(msg)

    # check whether connection to server was possible
    if status_code != 200:
        msg = f'Connection to server failed with status code {status_code}.'
        raise RuntimeError(msg)

    # handle the case where len(dataset) < max_workers
    max_workers = min(len(dataset), max_workers)

    # upload the samples
    if verbose:
        print(f'Uploading images (with {max_workers} workers).', flush=True)

    pbar = tqdm.tqdm(unit='imgs', total=len(dataset))
    tqdm_lock = tqdm.tqdm.get_lock()

    # define lambda function for concurrent upload
    def lambda_(i):
        # load image
        image, label, filename = dataset[i]
        # upload image
        success = _upload_single_image(
            image=image,
            label=label,
            filename=filename,
            dataset_id=dataset_id,
            token=token,
            mode=mode,
        )
        # update the progress bar
        tqdm_lock.acquire()  # lock
        pbar.update(1)  # update
        tqdm_lock.release()  # unlock
        # return whether the upload was successful
        return success

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda_, [i for i in range(len(dataset))], chunksize=1))

    if not all(results):
        msg = 'Warning: Unsuccessful upload(s)! '
        msg += 'This could cause problems when uploading embeddings.'
        msg += 'Failed at image: {}'.format(results.index(False))
        warnings.warn(msg)

    # set image type of data
    if mode == 'full':
        routes.users.datasets.put_image_type(dataset_id, token, mode)
    elif mode == 'thumbnails':
        routes.users.datasets.put_image_type(dataset_id, token, 'thumbnail')
    else:
        routes.users.datasets.put_image_type(dataset_id, token, 'meta')

    # create initial tag
    api_client = get_api_client(token=token)
    tags_api = TagsApi(api_client=api_client)

    initial_tag_create_request = InitialTagCreateRequest()
    tags_api.create_initial_tag_by_dataset_id(body=initial_tag_create_request, dataset_id=dataset_id)


def upload_images_from_folder(path_to_folder: str,
                              dataset_id: str,
                              token: str,
                              max_workers: int = 8,
                              mode: str = 'thumbnails',
                              size: int = -1,
                              verbose: bool = True):
    """Uploads images from a directory to the Lightly cloud solution.

    Args:
        path_to_folder:
            Path to the folder which holds the input images.
        dataset_id:
            The unique identifier for the dataset.
        token:
            Token for authentication.
        max_workers:
            Maximum number of workers uploading images in parallel.
        max_requests:
            Maximum number of requests a single worker can do before he has
            to wait for the others.
        mode:
            One of [full, thumbnails, metadata]. Whether to upload thumbnails,
            full images, or metadata only.
        size:
            Desired output size. If negative, default output size is used.
            If size is a sequence like (h, w), output size will be matched to 
            this. If size is an int, smaller edge of the image will be matched 
            to this number. i.e, if height > width, then image will be rescaled
            to (size * height / width, size).

    Raises:
        ValueError if dataset is too large.
        RuntimeError if the connection to the server failed.
        RuntimeError if dataset already has an initial tag.

    """

    transform = None
    if isinstance(size, tuple) or size > 0:
        transform = torchvision.transforms.Resize(size)

    dataset = LightlyDataset(input_dir=path_to_folder, transform=transform)
    upload_dataset(
        dataset,
        dataset_id,
        token,
        max_workers=max_workers,
        mode=mode,
        verbose=verbose,
    )


def _upload_metadata_from_json(path_to_embeddings: str, dataset_id: str, token: str):
    """TODO

    """
    msg = 'This site is under construction. Please come back later.'
    raise NotImplementedError(msg)


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
