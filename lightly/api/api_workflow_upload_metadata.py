from typing import Dict, List
from bisect import bisect_left

import tqdm

from lightly.openapi_generated.swagger_client.models.sample_update_request import \
    SampleUpdateRequest
from lightly.utils.io import COCO_ANNOTATION_KEYS


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
            KeyError if "images" or "metadata" aren't a key of custom_metadata.

        """
        _assert_key_exists_in_custom_metadata(
            COCO_ANNOTATION_KEYS.images, custom_metadata
        )
        _assert_key_exists_in_custom_metadata(
            COCO_ANNOTATION_KEYS.custom_metadata, custom_metadata
        )

    def index_custom_metadata_by_filename(self,
                                          filenames: List[str],
                                          custom_metadata: Dict):
        """Creates an index to lookup custom metadata by filename.

        Args:
            filenames:
                List of filenames.
            custom_metadata:
                Dictionary of custom metadata, see upload_custom_metadata for
                the required format.

        Returns:
            A dictionary containing custom metdata indexed by filename.

        """

        # sort images by filename
        custom_metadata[COCO_ANNOTATION_KEYS.images] = sorted(
            custom_metadata[COCO_ANNOTATION_KEYS.images],
            key=lambda x: x[COCO_ANNOTATION_KEYS.images_filename]
        )

        # sort metadata by image id
        custom_metadata[COCO_ANNOTATION_KEYS.custom_metadata] = sorted(
            custom_metadata[COCO_ANNOTATION_KEYS.custom_metadata],
            key=lambda x: x[COCO_ANNOTATION_KEYS.custom_metadata_image_id]
        )

        # get a list of filenames for binary search
        image_filenames = [
            image[COCO_ANNOTATION_KEYS.images_filename] for image in
            custom_metadata[COCO_ANNOTATION_KEYS.images]
        ]

        # get a list of image ids for binary search
        metadata_image_ids = [
            data[COCO_ANNOTATION_KEYS.custom_metadata_image_id] for data in
            custom_metadata[COCO_ANNOTATION_KEYS.custom_metadata]
        ]

        # map filename to metadata in O(n * logn)
        filename_to_metadata = {}
        for filename in filenames:

            image_index = bisect_left(image_filenames, filename)
            if image_index == len(image_filenames):
                raise RuntimeError(
                    f'Image with filename {filename} does not exist in custom metadata!'
                )

            image = custom_metadata[COCO_ANNOTATION_KEYS.images][image_index]
            image_id = image[COCO_ANNOTATION_KEYS.images_id]

            metadata_index = bisect_left(metadata_image_ids, image_id)
            if metadata_index == len(metadata_image_ids):
                raise RuntimeError(
                    f'Image with id {image_id} has no custom metadata!'
                )

            metadata = custom_metadata[COCO_ANNOTATION_KEYS.custom_metadata][
                metadata_index]
            filename_to_metadata[filename] = metadata

        return filename_to_metadata

    def upload_custom_metadata(self,
                               custom_metadata: Dict,
                               verbose: bool = False):
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

        """

        self.verify_custom_metadata_format(custom_metadata)

        # create a mapping from sample filenames to custom metadata
        samples = self.samples_api.get_samples_by_dataset_id(self.dataset_id)
        filename_to_metadata = self.index_custom_metadata_by_filename(
            [sample.file_name for sample in samples],
            custom_metadata,
        )

        if verbose:
            # wrap samples in a progress bar
            samples = tqdm.tqdm(samples)

        for sample in samples:

            metadata = filename_to_metadata[sample.file_name]

            if metadata is not None:
                # create a request to update the custom metadata of the sample
                update_sample_request = SampleUpdateRequest(
                    custom_meta_data=metadata
                )
                # send the request to the api
                self.samples_api.update_sample_by_id(
                    update_sample_request,
                    self.dataset_id,
                    sample.id
                )
