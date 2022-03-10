import concurrent
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union
from bisect import bisect_left

from tqdm import tqdm

from lightly.api.utils import retry
from lightly.cli._helpers import print_as_warning
from lightly.openapi_generated.swagger_client.models.sample_update_request import \
    SampleUpdateRequest
from lightly.utils.io import COCO_ANNOTATION_KEYS


class InvalidCustomMetadataWarning(Warning):
    pass


def _assert_key_exists_in_custom_metadata(key: str, dictionary: Dict):
    """Raises a formatted KeyError if key is not a key of the dictionary.
    
    """
    if key not in dictionary.keys():
        raise KeyError(
            f'Key {key} not found in custom metadata.\n'
            f'Found keys: {dictionary.keys()}'
        )


class _UploadCustomMetadataMixin:
    """Mixin of helpers to allow upload of custom metadata.

    """

    def verify_custom_metadata_format(self, custom_metadata: Dict):
        """Verifies that the custom metadata is in the correct format.

        Args:
            custom_metadata:
                Dictionary of custom metadata, see upload_custom_metadata for
                the required format.

        Raises:
            KeyError:
                If "images" or "metadata" aren't a key of custom_metadata.

        """
        _assert_key_exists_in_custom_metadata(
            COCO_ANNOTATION_KEYS.images, custom_metadata
        )
        _assert_key_exists_in_custom_metadata(
            COCO_ANNOTATION_KEYS.custom_metadata, custom_metadata
        )

    def index_custom_metadata_by_filename(self, custom_metadata: Dict)\
            -> Dict[str, Union[Dict, None]]:
        """Creates an index to lookup custom metadata by filename.

        Args:
            custom_metadata:
                Dictionary of custom metadata, see upload_custom_metadata for
                the required format.

        Returns:
            A dictionary mapping from filenames to custom metadata.
            If there are no annotations for a filename, the custom metadata
            is None instead.

        """

        # The mapping is filename -> image_id -> custom_metadata
        # This mapping is created in linear time.
        filename_to_image_id = {
            image_info[COCO_ANNOTATION_KEYS.images_filename]:
                image_info[COCO_ANNOTATION_KEYS.images_id]
            for image_info
            in custom_metadata[COCO_ANNOTATION_KEYS.images]
        }
        image_id_to_custom_metadata = {
            metadata[COCO_ANNOTATION_KEYS.custom_metadata_image_id]:
                metadata
            for metadata
            in custom_metadata[COCO_ANNOTATION_KEYS.custom_metadata]
        }
        filename_to_metadata = {
            filename: image_id_to_custom_metadata.get(image_id, None)
            for (filename, image_id)
            in filename_to_image_id.items()
        }
        return filename_to_metadata



    def upload_custom_metadata(self,
                               custom_metadata: Dict,
                               verbose: bool = False,
                               max_workers: int = 8):
        """Uploads custom metadata to the Lightly platform.

        The custom metadata is expected in a format similar to the COCO annotations:
        Under the key "images" there should be a list of dictionaries, each with
        a file_name and id. Under the key "metadata" the custom metadata is stored
        as a list of dictionaries, each with a image_id to match it to the image.

        Example:
            >>> custom_metadata = {
            >>>     "images": [
            >>>         {
            >>>             "file_name": "image0.jpg",
            >>>             "id": 0,
            >>>         },
            >>>         {
            >>>             "file_name": "image1.jpg",
            >>>             "id": 1,
            >>>         }
            >>>     ],
            >>>     "metadata": [
            >>>         {
            >>>             "image_id": 0,
            >>>             "number_of_people": 3,
            >>>             "weather": {
            >>>                 "scenario": "cloudy",
            >>>                 "temperature": 20.3
            >>>             }
            >>>         },
            >>>         {
            >>>             "image_id": 1,
            >>>             "number_of_people": 1,
            >>>             "weather": {
            >>>                 "scenario": "rainy",
            >>>                 "temperature": 15.0
            >>>             }
            >>>         }
            >>>     ]
            >>> }

        Args:
            custom_metadata:
                Custom metadata as described above.
            verbose:
                If True displays a progress bar during the upload.
            max_workers:
                Maximum number of concurrent threads during upload.

        """

        self.verify_custom_metadata_format(custom_metadata)



        # For each metadata, we need the corresponding sample_id
        # on the server. The mapping is:
        # metadata -> image_id -> filename -> sample_id

        image_id_to_filename = {
            image_info[COCO_ANNOTATION_KEYS.images_id]:
                image_info[COCO_ANNOTATION_KEYS.images_filename]
            for image_info in custom_metadata[COCO_ANNOTATION_KEYS.images]
        }

        samples = self._samples_api.get_samples_by_dataset_id(self.dataset_id)
        filename_to_sample_id = {
            sample.file_name: sample.id
            for sample in samples
        }

        upload_requests = []
        for metadata in custom_metadata[COCO_ANNOTATION_KEYS.custom_metadata]:
            image_id = metadata[COCO_ANNOTATION_KEYS.custom_metadata_image_id]
            filename = image_id_to_filename.get(image_id, None)
            if filename is None:
                print_as_warning(
                    f'No image found for custom metadata annotation '
                    f'with image_id {image_id}. '
                    f'This custom metadata annotation is skipped. '
                    f'Please fix your custom metadata file.',
                    InvalidCustomMetadataWarning
                )
                continue
            sample_id = filename_to_sample_id.get(filename, None)
            if sample_id is None:
                print_as_warning(
                    f'You tried to upload custom metadata for a sample with '
                    f'filename {{{filename}}}, '
                    f'but a sample with this filename '
                    f'does not exist on the server. '
                    f'This custom metadata annotation is skipped. '
                    f'Please upload the sample to the server first or fix '
                    f'your custom metadata file.',
                    InvalidCustomMetadataWarning
                )
                continue
            upload_request = (metadata, sample_id)
            upload_requests.append(upload_request)

        # retry upload if it times out
        def upload_sample_metadata(upload_request):
            metadata, sample_id = upload_request
            request = SampleUpdateRequest(custom_meta_data=metadata)
            return retry(
                self._samples_api.update_sample_by_id,
                request,
                dataset_id=self.dataset_id,
                sample_id=sample_id,
            )

        # Upload in parallel with a limit on the number of concurrent requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # get iterator over results
            results = executor.map(upload_sample_metadata, upload_requests)
            if verbose:
                results = tqdm(
                    results, 
                    unit='metadata',
                    total=len(upload_requests)
                )
            # iterate over results to make sure they are completed
            list(results)
