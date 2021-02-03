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
from lightly.api.utils import check_image
from lightly.api.utils import check_filename
from lightly.api.utils import PIL_to_bytes
from lightly.api.utils import put_request

from lightly.utils import fit_pca
from lightly.utils import load_embeddings_as_dict

from lightly.data import LightlyDataset

from lightly.openapi_generated.swagger_client.api_client import ApiClient
from lightly.openapi_generated.swagger_client.api.tags_api import TagsApi

import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor

import PIL.Image as Image


def _make_2d_embedding(batch, transformer):
    """Transforms a batch of embeddings to 2D.

    """
    # make a deepcopy of the batch
    batch_2d = copy.deepcopy(batch)

    # transform the embeddings to a 2-dimensional vector space
    for embedding in batch_2d['embeddings']:
        value_2d = transformer.transform(
            np.array(embedding['value'])[None, :]
        ).ravel()
        embedding['value'] = value_2d.tolist()

    return batch_2d


def _upload_single_batch(dataset_id,
                         token,
                         batch,
                         transformer,
                         append=False):
    """Uploads a batch of embeddings to the Lightly platform.

    """
    # indicate first batch
    batch['append'] = int(append)
    # how was the embedding generated?
    batch['type'] = 'model'
    # make 2d batch
    batch_2d = _make_2d_embedding(batch, transformer)
    # naming convention is name__2D
    batch_2d['embeddingName'] = batch_2d['embeddingName'] + '__2D'
    # which transform was applied?
    batch_2d['type'] = 'pca'
    # upload the embeddings
    routes.users.datasets.embeddings.post(dataset_id, token, batch)
    routes.users.datasets.embeddings.post(dataset_id, token, batch_2d)


def upload_embeddings_from_csv(path_to_embeddings: str,
                               dataset_id: str,
                               token: str,
                               max_upload: int = 32,
                               embedding_name: str = 'default',
                               verbose: bool = True):
    """Uploads embeddings from a csv file to the cloud solution.

    The csv file should be in the format specified by lightly. See the
    documentation on lightly.utils.io for more information.

    Args:
        path_to_embeddings:
            Path to csv file containing embeddings.
        dataset_id:
            The unique identifier for the dataset.
        token:
            Token for authentication.
        max_upload:
            Size of a batch of embeddings which is uploaded at once.
        embedding_name:
            Name the embedding will have on the Lightly web-app.

    Raises:
        RuntimeError if there is an error during the upload
        of a batch of embeddings.

    """
    data, embeddings, _, _ = load_embeddings_as_dict(
        path_to_embeddings,
        embedding_name,
        return_all=True,
    )

    data['token'] = token
    data['datasetId'] = dataset_id

    tags = routes.users.datasets.tags.get(dataset_id, token)
    if len(tags) == 0:
        msg = 'Forbidden upload to dataset with no existing tags.'
        raise RuntimeError(msg)

    embedding_summaries = routes.users.datasets.embeddings.get_summaries(
        dataset_id, token)
    embedding_names = [embedding['name'] for embedding in embedding_summaries]
    if embedding_name in embedding_names:
        msg = f'Forbidden upload: Embedding name {embedding_name} '
        msg += 'already exists.'
        raise RuntimeError(msg)

    # use pca to make 2d embeddings
    transformer = fit_pca(embeddings)
    n_embeddings = len(data['embeddings'])
    n_batches = n_embeddings // max_upload
    n_batches = n_batches + 1 if n_embeddings % max_upload else n_batches

    embedding_batches = [None] * n_batches
    for i in range(n_batches):
        left = i*max_upload
        right = min((i + 1) * max_upload, n_embeddings)
        batch = data.copy()
        batch['embeddings'] = data['embeddings'][left:right]
        embedding_batches[i] = batch

    if verbose:
        print('Uploading embeddings:')

    pbar = tqdm.tqdm(unit='embs', total=n_embeddings)
    for i, batch in enumerate(embedding_batches):
        _upload_single_batch(
            dataset_id, token, batch, transformer, append=i > 0)
        pbar.update(len(batch['embeddings']))


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

    try:
        sample_id = routes.users.datasets.samples.post(
            basename, thumbname, metadata, dataset_id, token
        )
    except RuntimeError:
        sample_upload_success = False

    # upload thumbnail
    thumbnail_upload_success = True
    if mode == 'thumbnails' and not metadata['is_corrupted']:
        try:
            # try to get signed url for thumbnail
            signed_url = routes.users.datasets.samples. \
                get_presigned_upload_url(
                    thumbname, dataset_id, sample_id, token)
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
            signed_url = routes.users.datasets.samples. \
                get_presigned_upload_url(
                    basename, dataset_id, sample_id, token)

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

    # check whether the dataset alreadys has existing tags
    tags = routes.users.datasets.tags.get(dataset_id, token)
    if len(tags) > 0:
        tag_names = [t['name'] for t in tags]
        msg = 'Forbidden upload to dataset with existing tags: '
        msg += f'{tag_names}'
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
        tqdm_lock.acquire() # lock
        pbar.update(1)      # update
        tqdm_lock.release() # unlock
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

    # set image type of data and create initial tag
    if mode == 'full':
        routes.users.datasets.put_image_type(dataset_id, token, mode)
    elif mode == 'thumbnails':
        routes.users.datasets.put_image_type(dataset_id, token, 'thumbnail')
    else:
        routes.users.datasets.put_image_type(dataset_id, token, 'meta')

    # create initial tag
    routes.users.datasets.tags.post(dataset_id, token)


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


def upload_csv(path_to_csv: str,
               dataset_id: str,
               token: str,
               name: Union[str, None] = None) -> str:
    """Requests a signed url and sends the CSV file there.

    Args:
        path_to_csv:
            Path to the csv file containing the embeddings to upload.
        dataset_id:
            The unique identifier for the dataset.
        token:
            Token for authentication.

    """
    # get a signed url for the csv file
    status, signed_url, embedding_id = routes.v1.datasets.embeddings.get_presigned_upload_url(
        dataset_id,
        token,
        name=name,
    )


    # upload the csv file using the signed url
    upload_file_with_signed_url(
        open(path_to_csv, 'rb'),
        signed_url,
    )

    return embedding_id


def _upload_metadata_from_json(path_to_embeddings: str,
                               dataset_id: str,
                               token: str):
    """TODO

    """
    msg = 'This site is under construction. Please come back later.'
    raise NotImplementedError(msg)


def upload_file_with_signed_url(file, url: str) -> bool:
    """Upload a file to the cloud storage using a signed URL.

    Args:
        filename:
            Path to a file for upload.
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


if __name__ == '__main__':

    path_to_csv = '/home/philipp_lightly_ai/lightly_outputs/2021-01-19/10-21-08/embeddings.csv'
    dataset_id = '5ff6fe276580b3000accaaa5'
    token = '347f1dfbc3879a142d536d0b'
    name = 'my-csv-2'

    upload_csv(
        path_to_csv,
        dataset_id,
        token,
        name=name
    )