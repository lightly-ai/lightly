from __future__ import annotations

import warnings
from concurrent.futures.thread import ThreadPoolExecutor
from typing import TYPE_CHECKING, Union

from lightly.openapi_generated.swagger_client.models.initial_tag_create_request import InitialTagCreateRequest
import tqdm

from lightly.api.upload import _upload_single_image

if TYPE_CHECKING:
    from lightly.api.api_workflow_client import ApiWorkflowClient

from lightly.api.constants import LIGHTLY_MAXIMUM_DATASET_SIZE
from lightly.data.dataset import LightlyDataset
from lightly.api.routes.users.service import get_quota


class _UploadDatasetMixin:

    def upload_dataset(self: ApiWorkflowClient, input: Union[str, LightlyDataset], max_workers: int = 8,
                       mode: str = 'thumbnails', verbose: bool = True):
        """Uploads a dataset to to the Lightly cloud solution.

        Args:
            input:
                one of the following:
                    - the path to the dataset, e.g. "path/to/dataset"
                    - the dataset in form of a LightlyDataset
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

        # Check input variable 'input'
        if isinstance(input, str):
            dataset = LightlyDataset(input_dir=input)
        elif isinstance(input, LightlyDataset):
            dataset = input
        else:
            raise ValueError(f"input must either be a LightlyDataset or the path to the dataset as str, "
                             f"but is of type {type(input)}")

        # check the allowed dataset size
        api_max_dataset_size, status_code = get_quota(self.token)
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
            # try to upload image
            try:
                _upload_single_image(
                    image=image,
                    label=label,
                    filename=filename,
                    dataset_id=self.dataset_id,
                    token=self.token,
                    mode=mode,
                )
                success = True
            except Exception as e:
                warnings.warn(f"Upload of image {filename} failed with error {e}")
                success = False

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

        # set image type of data and create initial tag
        if mode == 'full':
            img_type = 'full'
        elif mode == 'thumbnails':
            img_type = 'thumbnail'
        else:
            img_type = "meta"

        initial_tag_create_request = InitialTagCreateRequest(img_type=img_type)
        self.tags_api.create_initial_tag_by_dataset_id(body=initial_tag_create_request, dataset_id=self.dataset_id)
