import os
import warnings
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Union, List

import lightly_utils.image_processing
import tqdm

from lightly.openapi_generated.swagger_client import TagCreator, SamplesApi, SampleWriteUrls, SampleData
from lightly.openapi_generated.swagger_client.models.sample_create_request import SampleCreateRequest
from lightly.api.utils import check_filename, PIL_to_bytes
from lightly.openapi_generated.swagger_client.models.initial_tag_create_request import InitialTagCreateRequest
from lightly.data.dataset import LightlyDataset

from lightly.api.utils import retry

from lightly_utils import image_processing



class _UploadDatasetMixin:

    def upload_dataset(self, input: Union[str, LightlyDataset], max_workers: int = 8,
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
            ValueError if dataset is too large or input has the wrong type
            RuntimeError if the connection to the server failed.

        """
        no_tags_on_server = len(self._get_all_tags())
        if no_tags_on_server > 0:
            warnings.warn(f"Dataset with id {self.dataset_id} has already been completely uploaded to the platform. Skipping upload.")
            return

        # Check input variable 'input'
        if isinstance(input, str):
            dataset = LightlyDataset(input_dir=input)
        elif isinstance(input, LightlyDataset):
            dataset = input
        else:
            raise ValueError(f"input must either be a LightlyDataset or the path to the dataset as str, "
                             f"but is of type {type(input)}")

        # check the allowed dataset size
        max_dataset_size_str = self.quota_api.get_quota_maximum_dataset_size()
        max_dataset_size = int(max_dataset_size_str)
        if len(dataset) > max_dataset_size:
            msg = f'Your dataset has {len(dataset)} samples which'
            msg += f' is more than the allowed maximum of {max_dataset_size}'
            raise ValueError(msg)

        # handle the case where len(dataset) < max_workers
        max_workers = min(len(dataset), max_workers)
        max_workers = max(max_workers, 1)

        # upload the samples
        if verbose:
            print(f'Uploading images (with {max_workers} workers).', flush=True)


        # calculate the files size more efficiently
        lightly_utils.image_processing.metadata._size_in_bytes = lambda img: 0

        # get the filenames of the samples already on the server
        self.samples_api: SamplesApi
        samples: List[SampleData] = self.samples_api.get_samples_by_dataset_id(dataset_id=self.dataset_id)
        filenames = [sample.file_name for sample in samples]
        if len(filenames) > 0:
            print(f"Found {len(filenames)} images already on the server, they are skipped during the upload.")
        filenames_set = set(filenames)

        pbar = tqdm.tqdm(unit='imgs', total=len(dataset)-len(filenames))
        tqdm_lock = tqdm.tqdm.get_lock()

        # define lambda function for concurrent upload
        def lambda_(i):
            # load image
            image, label, filename = dataset[i]
            if filename in filenames_set:
                return True

            filepath = dataset.get_filepath_from_filename(filename, image)
            # try to upload image
            try:
                self._upload_single_image(
                    image=image,
                    label=label,
                    filename=filename,
                    filepath=filepath,
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

        initial_tag_create_request = InitialTagCreateRequest(img_type=img_type, creator=TagCreator.USER_PIP)
        self.tags_api.create_initial_tag_by_dataset_id(body=initial_tag_create_request, dataset_id=self.dataset_id)

    def _upload_single_image(self, image, label, filename: str, filepath: str, mode):
        """Uploads a single image to the Lightly platform.

        """
        self.samples_api: SamplesApi

        # check whether the filename is too long
        basename = filename
        if not check_filename(basename):
            msg = (f'Filename {basename} is longer than the allowed maximum of '
                   'characters and will be skipped.')
            warnings.warn(msg)
            return False

        # calculate metadata, and check if corrupted
        metadata = image_processing.Metadata(image).to_dict()
        metadata["sizeInBytes"] = os.path.getsize(filepath)

        # try to get exif data
        try:
            exifdata = image_processing.Exifdata(image)
        except Exception:
            exifdata = None

        # generate thumbnail if necessary
        thumbname = None
        if not metadata['is_corrupted'] and mode in ["thumbnails", "full"]:
            thumbname = '.'.join(basename.split('.')[:-1]) + '_thumb.webp'

        body = SampleCreateRequest(
            file_name=basename,
            thumb_name=thumbname,
            meta_data=metadata,
            exif=exifdata if exifdata is None else exifdata.to_dict(),
        )
        sample_id = retry(
            self.samples_api.create_sample_by_dataset_id,
            body=body,
            dataset_id=self.dataset_id
        ).id

        if not metadata['is_corrupted'] and mode in ["thumbnails", "full"]:

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

            if mode == "thumbnails":
                thumbnail_url = retry(
                    self.samples_api.get_sample_image_write_url_by_id,
                    dataset_id=self.dataset_id,
                    sample_id=sample_id,
                    is_thumbnail=True
                )
                upload_thumbnail(image, thumbnail_url)
            elif mode == "full":
                sample_write_urls: SampleWriteUrls = retry(
                    self.samples_api.get_sample_image_write_urls_by_id, dataset_id=self.dataset_id, sample_id=sample_id
                )
                upload_thumbnail(image, sample_write_urls.thumb)
                upload_full_image(filepath, sample_write_urls.full)



        image.close()


