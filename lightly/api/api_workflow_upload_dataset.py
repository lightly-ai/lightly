from __future__ import annotations

import warnings
from concurrent.futures.thread import ThreadPoolExecutor
from typing import TYPE_CHECKING, Union

import torchvision

from lightly.openapi_generated.swagger_client.models.sample_create_request import SampleCreateRequest

from lightly.api.utils import check_filename, check_image, get_thumbnail_from_img, PIL_to_bytes

from lightly.openapi_generated.swagger_client.models.initial_tag_create_request import InitialTagCreateRequest
import tqdm

if TYPE_CHECKING:
    from lightly.api.api_workflow_client import ApiWorkflowClient

from lightly.api.constants import LIGHTLY_MAXIMUM_DATASET_SIZE
from lightly.data.dataset import LightlyDataset
from lightly.api.routes.users.service import get_quota


class _UploadDatasetMixin:

    def upload_dataset(self: ApiWorkflowClient, input: Union[str, LightlyDataset], max_workers: int = 8,
                       size: Union[int, tuple] = -1,  mode: str = 'thumbnails', verbose: bool = True):
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
            size:
                Desired output size. If negative, default output size is used.
                If size is a sequence like (h, w), output size will be matched to 
                this. If size is an int, smaller edge of the image will be matched 
                to this number. i.e, if height > width, then image will be rescaled
                to (size * height / width, size).
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
            transform = None
            if isinstance(size, tuple) or size > 0:
                transform = torchvision.transforms.Resize(size)
            dataset = LightlyDataset(input_dir=input, transform=transform)
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
                self._upload_single_image(
                    image=image,
                    label=label,
                    filename=filename,
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
            img_type = 'meta'

        initial_tag_create_request = InitialTagCreateRequest(img_type=img_type)
        self.tags_api.create_initial_tag_by_dataset_id(body=initial_tag_create_request, dataset_id=self.dataset_id)

    def _upload_single_image(self: ApiWorkflowClient, image, label, filename: str, mode):
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
        if not metadata['is_corrupted'] and mode in ["thumbnails", "full"]:
            thumbname = '.'.join(basename.split('.')[:-1]) + '_thumb.webp'

        body = SampleCreateRequest(file_name=basename, thumb_name=thumbname, meta_data=metadata)
        sample_id = self.samples_api.create_sample_by_dataset_id(body=body, dataset_id=self.dataset_id).id

        if not metadata['is_corrupted'] and mode in ["thumbnails", "full"]:
            thumbnail = get_thumbnail_from_img(image)
            image_to_upload = PIL_to_bytes(thumbnail, ext='webp', quality=90)

            signed_url = self.samples_api.get_sample_image_write_url_by_id(
                dataset_id=self.dataset_id, sample_id=sample_id, is_thumbnail=True)

            # try to upload thumbnail
            self.upload_file_with_signed_url(image_to_upload, signed_url)

            thumbnail.close()

        if not metadata['is_corrupted'] and mode == "full":
            image_to_upload = PIL_to_bytes(image)

            signed_url = self.samples_api.get_sample_image_write_url_by_id(
                dataset_id=self.dataset_id, sample_id=sample_id, is_thumbnail=False)

            # try to upload thumbnail
            self.upload_file_with_signed_url(image_to_upload, signed_url)

            image.close()


