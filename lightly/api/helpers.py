""" Communication Helpers """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
import copy
import time
import random

import numpy as np

from itertools import islice
from typing import List

from lightly.api.communication import create_initial_tag
from lightly.api.communication import get_presigned_upload_url
from lightly.api.communication import upload_file_with_signed_url
from lightly.api.communication import upload_sample_with_metadata
from lightly.api.communication import upload_embedding
from lightly.api.communication import put_image_type
from lightly.api.communication import get_samples
from lightly.api.communication import get_tags
from lightly.api.communication import get_user_quota
from lightly.api.communication import get_embedding_summaries

from lightly.api.utils import get_thumbnail_from_img
from lightly.api.utils import check_image
from lightly.api.utils import PIL_to_bytes

from lightly.data import LightlyDataset
from lightly.utils import fit_pca
from lightly.utils import load_embeddings_as_dict

import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor

import PIL.Image as Image

MAXIMUM_DATASET_SIZE = 25_000


def upload_embeddings_from_csv(path_to_embeddings: str,
                               dataset_id: str,
                               token: str,
                               max_upload: int = 32,
                               embedding_name: str = 'default'):
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

    tags = get_tags(dataset_id, token)
    if len(tags) == 0:
        msg = 'Forbidden upload to dataset with no existing tags.'
        raise RuntimeError(msg)

    embedding_summaries = get_embedding_summaries(dataset_id, token)
    embedding_names = [embedding['name'] for embedding in embedding_summaries]
    if embedding_name in embedding_names:
        msg = f'Forbidden upload: Embedding name {embedding_name} '
        msg += 'already exists.'
        raise RuntimeError(msg)

    # use pca to make 2d embeddings
    transformer = fit_pca(embeddings)

    def _make_2d_embedding(batch):
        batch_2d = copy.deepcopy(batch)
        for embedding in batch_2d['embeddings']:
            value_2d = transformer.transform(
                np.array(embedding['value'])[None, :]
            ).ravel()
            embedding['value'] = value_2d.tolist()
        return batch_2d

    def _upload_single_batch(batch, append=False):
        batch['append'] = int(append)
        batch['type'] = 'model'
        batch_2d = _make_2d_embedding(batch)
        batch_2d['embeddingName'] = batch_2d['embeddingName'] + '__2D'
        batch_2d['type'] = 'pca'
        upload_embedding(batch)
        upload_embedding(batch_2d)

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

    pbar = tqdm.tqdm(unit='embs', total=n_embeddings)
    for i, batch in enumerate(embedding_batches):
        _upload_single_batch(batch, append=i > 0)
        pbar.update(len(batch['embeddings']))


def upload_images_from_folder(path_to_folder: str,
                              dataset_id: str,
                              token: str,
                              max_workers: int = 8,
                              max_requests: int = 32,
                              mode: str = 'thumbnails'):
    """Uploads images from a directory to the Lightly cloud solution.

    Args:
        path_to_folder:
            Path to the folder containing the images.
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
        RuntimeError if dataset already has an initial tag.

    """

    bds = LightlyDataset(from_folder=path_to_folder)
    fnames = bds.get_filenames()

    api_max_dataset_size = get_user_quota(token)['maxDatasetSize']
    max_dataset_size = min(api_max_dataset_size, MAXIMUM_DATASET_SIZE)

    if len(fnames) > max_dataset_size:
        msg = f'Your dataset has {len(fnames)} samples which'
        msg += f' is more than the allowed maximum of {max_dataset_size}'
        raise ValueError(msg)

    tags = get_tags(dataset_id, token)
    if len(tags) > 0:
        tag_names = [t['name'] for t in tags]
        msg = 'Forbidden upload to dataset with existing tags: '
        msg += f'{tag_names}'
        raise RuntimeError(msg)

    def _upload_single_image(fname):

        # random delay of uniform[0, 0.01] seconds to prevent API bursts
        rnd_delay = random.random() * 0.01
        time.sleep(rnd_delay)

        # get PIL image handles, metadata, and check if corrupted
        metadata, is_corrupted = check_image(
            os.path.join(path_to_folder, fname)
        )

        # filename is too long, cannot accept this file
        if not metadata:
            return False

        # upload sample
        basename = fname
        thumbname = None
        if mode in ['full', 'thumbnails'] and not is_corrupted:
            thumbname = '.'.join(basename.split('.')[:-1]) + '_thumb.webp'

        sample_upload_success = True
        try:
            sample_id = upload_sample_with_metadata(
                basename, thumbname, metadata, dataset_id, token
            )
        except RuntimeError:
            sample_upload_success = False

        # upload thumbnail
        thumbnail_upload_success = True
        if mode == 'thumbnails' and not is_corrupted:
            try:
                # try to get signed url for thumbnail
                signed_url = get_presigned_upload_url(
                    thumbname, dataset_id, sample_id, token
                )
                # try to create thumbnail
                image_path = os.path.join(path_to_folder, fname)
                with Image.open(image_path) as temp_image:
                    thumbnail = get_thumbnail_from_img(temp_image)
                # try to upload thumbnail
                upload_file_with_signed_url(
                    PIL_to_bytes(thumbnail, ext='webp', quality=70),
                    signed_url
                )
            except RuntimeError:
                thumbnail_upload_success = False

        # upload full image
        image_upload_success = True
        if mode == 'full' and not is_corrupted:
            try:
                # try to get signed url for image
                signed_url = get_presigned_upload_url(
                    basename, dataset_id, sample_id, token
                )

                # try to upload image
                image_path = os.path.join(path_to_folder, fname)
                with open(image_path, 'rb') as temp_image:
                    upload_file_with_signed_url(
                        temp_image,
                        signed_url
                    )
            except RuntimeError:
                image_upload_success = False

        success = sample_upload_success
        success = success and thumbnail_upload_success
        success = success and image_upload_success
        return success

    n_batches = len(fnames) // max_requests
    n_batches = n_batches + 1 if len(fnames) % max_requests else n_batches
    fname_batches = [
        list(islice(fnames, i * max_requests, (i + 1) * max_requests))
        for i in range(n_batches)
    ]

    chunksize = max(max_requests // max_workers, 1)
    executor = ThreadPoolExecutor(max_workers=max_workers)

    pbar = tqdm.tqdm(unit='imgs', total=len(fnames))
    for i, batch in enumerate(fname_batches):
        mapped = executor.map(_upload_single_image, batch, chunksize=chunksize)
        mapped = list(mapped)
        if not all(mapped):
            msg = 'Warning: Unsuccessful upload(s) in batch {}! '.format(i)
            msg += 'This could cause problems when uploading embeddings.'
            msg += 'Failed at file: {}'.format(mapped.index(False))
            warnings.warn(msg)
        pbar.update(len(batch))

    # set image type of data and create initial tag
    if mode == 'full':
        put_image_type(dataset_id, token, mode)
    elif mode == 'thumbnails':
        put_image_type(dataset_id, token, 'thumbnail')
    else:
        put_image_type(dataset_id, token, 'meta')
    create_initial_tag(dataset_id, token)


def upload_metadata_from_json(path_to_embeddings: str,
                              dataset_id: str,
                              token: str):
    pass


def get_samples_by_tag(tag_name: str,
                       dataset_id: str,
                       token: str,
                       mode: str = 'list',
                       filenames: List[str] = None):
    """Get the files associated with a given tag and dataset.

    Asks the servers for all samples in a given tag and dataset. If mode is
    mask or indices, the list of all filenames must be specified. Can return
    either the list of all filenames in the tag, or a mask or indices 
    indicating which of the provided filenames are in the tag.

    Args:
        tag_name:
            Name of the tag to query.
        dataset_id:
            The unique identifier of the dataset.
        token:
            Token for authentication.
        mode:
            Return type, must be in ["list", "mask", "indices"].
        filenames:
            List of all filenames.

    Returns:
        Either list of filenames, binary mask, or list of indices
        specifying the samples in the requested tag.

    Raises:
        ValueError, RuntimeError

    """

    if mode == 'mask' and filenames is None:
        msg = f'Argument filenames must not be None for mode "{mode}"!'
        raise ValueError(msg)
    if mode == 'indices' and filenames is None:
        msg = f'Argument filenames must not be None for mode "{mode}"!'
        raise ValueError(msg)

    samples = get_samples(dataset_id, token, tag_name=tag_name)

    if mode == 'list':
        return samples

    if mode == 'mask':
        mask = [1 if f in set(samples) else 0 for f in filenames]
        if sum(mask) != len(samples):
            msg = 'Error during mapping from samples to filenames: '
            msg += f'sum(mask) != len(samples) with lengths '
            msg += f'{sum(mask)} and {len(samples)}'
            raise RuntimeError(msg)
        return mask

    if mode == 'indices':
        indices = [i for i in range(len(filenames))]
        indices = filter(lambda i: filenames[i] in set(samples), indices)
        indices = list(indices)
        if len(indices) != len(samples):
            msg = 'Error during mapping from samples to filenames: '
            msg += f'len(indices) != len(samples) with lengths '
            msg += f'{len(indices)} and {len(samples)}.'
            raise RuntimeError(msg)
        return indices

    msg = f'Got illegal mode "{mode}"! '
    msg += 'Must be in ["list", "mask", "indices"]'
    raise ValueError(msg)
