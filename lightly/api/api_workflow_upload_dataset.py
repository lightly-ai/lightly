""" Upload Dataset Mixin """

import os
import warnings
from typing import Union, Dict
from datetime import datetime
from concurrent.futures.thread import ThreadPoolExecutor

import tqdm
from lightly_utils import image_processing

from lightly.data.dataset import LightlyDataset
from lightly.api.utils import check_filename
from lightly.api.utils import MAXIMUM_FILENAME_LENGTH
from lightly.api.utils import retry
from lightly.openapi_generated.swagger_client import TagCreator
from lightly.openapi_generated.swagger_client import SampleWriteUrls
from lightly.openapi_generated.swagger_client.models.sample_create_request \
    import SampleCreateRequest
from lightly.openapi_generated.swagger_client.models.tag_upsize_request \
    import TagUpsizeRequest
from lightly.openapi_generated.swagger_client.models.initial_tag_create_request\
    import InitialTagCreateRequest
from lightly.openapi_generated.swagger_client.models.job_status_meta \
    import JobStatusMeta
from lightly.openapi_generated.swagger_client.models.job_status_upload_method \
    import JobStatusUploadMethod


class _UploadDatasetMixin:
    """Mixin to upload datasets to the Lightly Api.

    """

    def upload_dataset(self,
                       input: Union[str, LightlyDataset],
                       max_workers: int = 8,
                       mode: str = 'thumbnails',
                       verbose: bool = True,
                       custom_metadata: Union[Dict, None] = None):
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
                One of [full, thumbnails, metadata]. Whether to upload
                thumbnails, full images, or metadata only.

        Raises:
            ValueError if dataset is too large or input has the wrong type
            RuntimeError if the connection to the server failed.

        """

        # get all tags of the dataset
        tags = self._get_all_tags()
        if len(tags) > 0:
            print(
                f'Dataset with id {self.dataset_id} has {len(tags)} tags.',
                flush=True
            )

        # parse "input" variable
        if isinstance(input, str):
            dataset = LightlyDataset(input_dir=input)
        elif isinstance(input, LightlyDataset):
            dataset = input
        else:
            raise ValueError(
                f'input must either be a LightlyDataset or the path to the'
                f'dataset as str, but has type {type(input)}'
            )

        # handle the case where len(dataset) < max_workers
        max_workers = min(len(dataset), max_workers)
        max_workers = max(max_workers, 1)

        # upload the samples
        if verbose:
            print(
                f'Uploading images (with {max_workers} workers).',
                flush=True
            )

        # TODO: remove _size_in_bytes from image_processing
        image_processing.metadata._size_in_bytes = \
            lambda img: 0 # pylint: disable=protected-access

        # get the filenames of the samples already on the server
        samples = self.samples_api.get_samples_by_dataset_id(
            dataset_id=self.dataset_id
        )
        filenames_on_server = [sample.file_name for sample in samples]
        filenames_on_server_set = set(filenames_on_server)
        if len(filenames_on_server) > 0:
            print(
                f'Found {len(filenames_on_server)} images already on the server'
                ', they are skipped during the upload.'
            )

        # check the maximum allowed dataset size
        total_filenames = set(dataset.get_filenames()).union(
            filenames_on_server_set
        )
        max_dataset_size = \
            int(self.quota_api.get_quota_maximum_dataset_size())
        if len(total_filenames) > max_dataset_size:
            msg = f'Your dataset has {len(dataset)} samples which'
            msg += f' is more than the allowed maximum of {max_dataset_size}'
            raise ValueError(msg)

        # index custom metadata by filename (only if it exists)
        filename_to_metadata = {}
        if custom_metadata is not None:
            self.verify_custom_metadata_format(custom_metadata)
            filename_to_metadata = self.index_custom_metadata_by_filename(
                dataset.get_filenames(),
                custom_metadata,
            )

        # register dataset upload
        job_status_meta = JobStatusMeta(
            total=len(total_filenames),
            processed=len(filenames_on_server),
            is_registered=True,
            upload_method=JobStatusUploadMethod.USER_PIP,
        )
        self.datasets_api.register_dataset_upload_by_id(
            job_status_meta,
            self.dataset_id
        )

        pbar = tqdm.tqdm(
            unit='imgs',
            total=len(total_filenames) - len(filenames_on_server),
        )
        tqdm_lock = tqdm.tqdm.get_lock()

        # define lambda function for concurrent upload
        def lambda_(i):
            # load image
            image, _, filename = dataset[i]
            if filename in filenames_on_server_set:
                # the sample was already uploaded
                return True

            filepath = dataset.get_filepath_from_filename(filename, image)

            # get custom metadata (evaluates to None if there is none)
            custom_metadata_item = filename_to_metadata.get(filename, None)

            # try to upload image
            try:
                self._upload_single_image(
                    image=image,
                    filename=filename,
                    filepath=filepath,
                    mode=mode,
                    custom_metadata=custom_metadata_item,
                )
                success = True
            except Exception as e: # pylint: disable=broad-except
                warnings.warn(
                    f'Upload of image {filename} failed with error {e}'
                )
                success = False

            # update the progress bar
            tqdm_lock.acquire()
            pbar.update(1)
            tqdm_lock.release()
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

        if len(tags) == 0:
            # create initial tag
            initial_tag_create_request = InitialTagCreateRequest(
                img_type=img_type,
                creator=TagCreator.USER_PIP
            )
            self.tags_api.create_initial_tag_by_dataset_id(
                body=initial_tag_create_request,
                dataset_id=self.dataset_id,
            )
        else:
            # upsize existing tags
            upsize_tags_request = TagUpsizeRequest(
                upsize_tag_name=datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss'),
                upsize_tag_creator=TagCreator.USER_PIP,
            )
            self.tags_api.upsize_tags_by_dataset_id(
                body=upsize_tags_request,
                dataset_id=self.dataset_id,
            )

    def _upload_single_image(self,
                             image,
                             filename: str,
                             filepath: str,
                             mode: str,
                             custom_metadata: Union[Dict, None] = None):
        """Uploads a single image to the Lightly platform.

        """
        # check whether the filepath is too long
        if not check_filename(filepath):
            msg = ('Filepath {filepath} is longer than the allowed maximum of '
                   f'{MAXIMUM_FILENAME_LENGTH} characters and will be skipped.')
            raise ValueError(msg)

        # calculate metadata, and check if corrupted
        metadata = image_processing.Metadata(image).to_dict()
        metadata['sizeInBytes'] = os.path.getsize(filepath)

        # try to get exif data
        try:
            exifdata = image_processing.Exifdata(image)
        except Exception: # pylint disable=broad-except
            exifdata = None

        # generate thumbnail if necessary
        thumbname = None
        if not metadata['is_corrupted'] and mode in ['thumbnails', 'full']:
            thumbname = '.'.join(filename.split('.')[:-1]) + '_thumb.webp'

        body = SampleCreateRequest(
            file_name=filename,
            thumb_name=thumbname,
            meta_data=metadata,
            exif=exifdata if exifdata is None else exifdata.to_dict(),
            custom_meta_data=custom_metadata,
        )
        sample_id = retry(
            self.samples_api.create_sample_by_dataset_id,
            body=body,
            dataset_id=self.dataset_id
        ).id

        if not metadata['is_corrupted'] and mode in ['thumbnails', 'full']:

            def upload_thumbnail(image, signed_url):
                thumbnail = image_processing.Thumbnail(image)
                image_to_upload = thumbnail.to_bytes()
                retry(
                    self.upload_file_with_signed_url,
                    image_to_upload,
                    signed_url
                )
                thumbnail.thumbnail.close()

            def upload_full_image(filepath, signed_url):
                with open(filepath, 'rb') as image_to_upload:
                    retry(
                        self.upload_file_with_signed_url,
                        image_to_upload,
                        signed_url
                    )

            if mode == 'thumbnails':
                thumbnail_url = retry(
                    self.samples_api.get_sample_image_write_url_by_id,
                    dataset_id=self.dataset_id,
                    sample_id=sample_id,
                    is_thumbnail=True
                )
                upload_thumbnail(image, thumbnail_url)
            elif mode == 'full':
                sample_write_urls: SampleWriteUrls = retry(
                    self.samples_api.get_sample_image_write_urls_by_id,
                    dataset_id=self.dataset_id,
                    sample_id=sample_id
                )
                upload_thumbnail(image, sample_write_urls.thumb)
                upload_full_image(filepath, sample_write_urls.full)

        image.close()
