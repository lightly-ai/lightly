from typing import Dict, List
from bisect import bisect_left

import tqdm

from lightly.openapi_generated.swagger_client.models.sample_update_request import SampleUpdateRequest


class _COCO_ANNOTATION_KEYS:
    """TODO
    
    """
    # TODO
    images: str = 'images'
    images_id: str = 'id'
    images_filename: str = 'file_name'

    # TODO
    custom_metadata: str = 'metadata'
    custom_metadata_image_id: str = 'image_id'


class _UploadCustomMetadataMixin:
    """TODO
    
    """

    def verify_custom_metadata_format(self, custom_metadata: Dict):
        pass


    def index_custom_metadata_by_filename(self,
                                          filenames: List[str],
                                          custom_metadata: Dict):
        """TODO
        
        """

        # sort images by filename
        custom_metadata[_COCO_ANNOTATION_KEYS.images] = sorted(
            custom_metadata[_COCO_ANNOTATION_KEYS.images],
            key=lambda x: x[_COCO_ANNOTATION_KEYS.images_filename]
        )

        # sort metadata by image id
        custom_metadata[_COCO_ANNOTATION_KEYS.custom_metadata] = sorted(
            custom_metadata[_COCO_ANNOTATION_KEYS.custom_metadata],
            key=lambda x: x[_COCO_ANNOTATION_KEYS.custom_metadata_image_id]
        )

        # TODO
        image_filenames = [
            image[_COCO_ANNOTATION_KEYS.images_filename] for image in 
            custom_metadata[_COCO_ANNOTATION_KEYS.images]        
        ]

        # TODO
        metadata_image_ids = [
            data[_COCO_ANNOTATION_KEYS.custom_metadata_image_id] for data in 
            custom_metadata[_COCO_ANNOTATION_KEYS.custom_metadata]        
        ]

        # map filename to metadata in O(n * logn)
        filename_to_metadata = {}
        for filename in filenames:

            image_index = bisect_left(image_filenames, filename)
            if image_index == len(image_filenames):
                # didn't find an image with that filename
                raise RuntimeError('TODO')

            image = custom_metadata[_COCO_ANNOTATION_KEYS.images][image_index]
            image_id = image[_COCO_ANNOTATION_KEYS.images_id]

            metadata_index = bisect_left(metadata_image_ids, image_id)
            if metadata_index == len(metadata_image_ids):
                # didn't find custom metadata with for this image
                raise RuntimeError('TODO')

            metadata = custom_metadata[_COCO_ANNOTATION_KEYS.custom_metadata][metadata_index]
            filename_to_metadata[filename] = metadata

        return filename_to_metadata


    def upload_custom_metadata(self,
                               custom_metadata: Dict,
                               verbose: bool = False):
        """TODO
        
        """

        self.verify_custom_metadata_format(custom_metadata)

        # TODO
        samples = self.samples_api.get_samples_by_dataset_id(self.dataset_id)
        filename_to_metadata = self.index_custom_metadata_by_filename(
            [sample.file_name for sample in samples],
            custom_metadata,
        )

        if verbose:
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
