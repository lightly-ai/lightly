import warnings
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Union
import io
import os
import tqdm
from urllib.request import Request, urlopen
from PIL import Image

from lightly.openapi_generated.swagger_client import TagCreator
from lightly.openapi_generated.swagger_client.models.sample_create_request import SampleCreateRequest
from lightly.api.utils import check_filename, check_image, get_thumbnail_from_img, PIL_to_bytes
from lightly.api.bitmask import BitMask
from lightly.openapi_generated.swagger_client.models.initial_tag_create_request import InitialTagCreateRequest
from lightly.openapi_generated.swagger_client.models.image_type import ImageType
from lightly.data.dataset import LightlyDataset



def _make_dir_and_save_image(output_dir: str, filename: str, img: Image):
    """

    """
    path = os.path.join(output_dir, filename)

    head = os.path.split(path)[0]
    if not os.path.exists(head):
        os.makedirs(head)

    img.save(path)


class _DownloadDatasetMixin:

    def download_dataset(self,
                         output_dir: str,
                         tag_name: str = 'initial-tag',
                         verbose: bool = True):
        """TODO

        Args:
            tag_name:
                Name of the tag which should be downloaded.
            verbose:
                TODO

        Raises:
            ValueError if the specified tag does not exist on the dataset.
            RuntimeError if the connection to the server failed.

        """

        # check if images are available
        dataset = self.datasets_api.get_dataset_by_id(self.dataset_id)
        if dataset.img_type != ImageType.FULL:
            # only thumbnails or metadata available
            raise ValueError(
                f"Dataset with id {self.dataset_id} has no downloadable images!"
            )

        # check if tag exists
        available_tags = self._get_all_tags()
        try:
            tag = next(tag for tag in available_tags if tag.name == tag_name)
        except StopIteration:
            tag = None

        if tag is None:
            raise ValueError(
                f"Dataset with id {self.dataset_id} has no tag {tag_name}!"
            )

        # get sample ids
        sample_ids = self.mappings_api.get_sample_mappings_by_dataset_id(
            self.dataset_id,
            field='_id'
        )

        indices = BitMask.from_hex(tag.bit_mask_data).to_indices()
        sample_ids = [sample_ids[i] for i in indices]
        filenames = [self.filenames_on_server[i] for i in indices]

        if verbose:
            print(f'Downloading {len(sample_ids)} images:', flush=True)
            pbar = tqdm.tqdm(unit='imgs', total=len(sample_ids))

        # download images
        for sample_id, filename in zip(sample_ids, filenames):
            read_url = self.samples_api.get_sample_image_read_url_by_id(
                self.dataset_id, 
                sample_id,
                type="full",
            )
            request = Request(read_url, method='GET')
            with urlopen(request) as response:
                blob = response.read()
                img = Image.open(io.BytesIO(blob))
                _make_dir_and_save_image(output_dir, filename, img)
            
            if verbose:
                pbar.update(1)
