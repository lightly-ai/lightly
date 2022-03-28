from typing import Dict, List
import io
import os
import tqdm
from urllib.request import Request, urlopen
from PIL import Image

from lightly.api.bitmask import BitMask
from lightly.openapi_generated.swagger_client.models.image_type import ImageType



def _make_dir_and_save_image(output_dir: str, filename: str, img: Image):
    """Saves the images and creates necessary subdirectories.

    """
    path = os.path.join(output_dir, filename)

    head = os.path.split(path)[0]
    if not os.path.exists(head):
        os.makedirs(head)

    img.save(path)
    img.close()


def _get_image_from_read_url(read_url: str):
    """Makes a get request to the signed read url and returns the image.

    """
    request = Request(read_url, method='GET')
    with urlopen(request) as response:
        blob = response.read()
        img = Image.open(io.BytesIO(blob))
    return img


class _DownloadDatasetMixin:

    def download_dataset(self,
                         output_dir: str,
                         tag_name: str = 'initial-tag',
                         verbose: bool = True):
        """Downloads images from the web-app and stores them in output_dir.

        Args:
            output_dir:
                Where to store the downloaded images.
            tag_name:
                Name of the tag which should be downloaded.
            verbose:
                Whether or not to show the progress bar.

        Raises:
            ValueError:
                If the specified tag does not exist on the dataset.
            RuntimeError:
                If the connection to the server failed.

        """

        # check if images are available
        dataset = self._datasets_api.get_dataset_by_id(self.dataset_id)
        if dataset.img_type != ImageType.FULL:
            # only thumbnails or metadata available
            raise ValueError(
                f"Dataset with id {self.dataset_id} has no downloadable images!"
            )

        # check if tag exists
        available_tags = self.get_all_tags()
        try:
            tag = next(tag for tag in available_tags if tag.name == tag_name)
        except StopIteration:
            raise ValueError(
                f"Dataset with id {self.dataset_id} has no tag {tag_name}!"
            )

        # get sample ids
        sample_ids = self._mappings_api.get_sample_mappings_by_dataset_id(
            self.dataset_id,
            field='_id'
        )

        indices = BitMask.from_hex(tag.bit_mask_data).to_indices()
        sample_ids = [sample_ids[i] for i in indices]
        filenames_on_server = self.get_filenames()
        filenames = [filenames_on_server[i] for i in indices]

        if verbose:
            print(f'Downloading {len(sample_ids)} images:', flush=True)
            pbar = tqdm.tqdm(unit='imgs', total=len(sample_ids))

        # download images
        for sample_id, filename in zip(sample_ids, filenames):
            read_url = self._samples_api.get_sample_image_read_url_by_id(
                self.dataset_id, 
                sample_id,
                type="full",
            )

            img = _get_image_from_read_url(read_url)
            _make_dir_and_save_image(output_dir, filename, img)

            if verbose:
                pbar.update(1)

    def export_label_studio_tasks_by_tag_id(
        self,
        tag_id: str,
    ) -> List[Dict]:
        """Exports samples in a format compatible with Label Studio.

        The format is documented here:
        https://labelstud.io/guide/tasks.html#Basic-Label-Studio-JSON-format

        Args:
            tag_id:
                Id of the tag which should exported.

        Returns:
            A list of dictionaries in a format compatible with Label Studio.

        """
        label_studio_tasks = self._tags_api.export_tag_to_label_studio_tasks(
            self.dataset_id,
            tag_id
        )
        return label_studio_tasks

    def export_label_studio_tasks_by_tag_name(
        self,
        tag_name: str,
    ) -> List[Dict]:
        """Exports samples in a format compatible with Label Studio.

        The format is documented here:
        https://labelstud.io/guide/tasks.html#Basic-Label-Studio-JSON-format

        Args:
            tag_name:
                Name of the tag which should exported.

        Returns:
            A list of dictionaries in a format compatible with Label Studio.

        Examples:
            >>> # write json file which can be imported in Label Studio
            >>> tasks = client.export_label_studio_tasks_by_tag_name(
            >>>     'initial-tag'
            >>> )
            >>> 
            >>> with open('my-label-studio-tasks.json', 'w') as f:
            >>>     json.dump(tasks, f)

        """
        tag = self.get_tag_by_name(tag_name)
        return self.export_label_studio_tasks_by_tag_id(tag.id)

    def export_label_box_data_rows_by_tag_id(
        self,
        tag_id: str,
    ) -> List[Dict]:
        """Exports samples in a format compatible with Labelbox.

        The format is documented here:
        https://docs.labelbox.com/docs/images-json

        Args:
            tag_id:
                Id of the tag which should exported.

        Returns:
            A list of dictionaries in a format compatible with Labelbox.

        """
        label_box_data_rows = self._tags_api.export_tag_to_label_box_data_rows(
            self.dataset_id,
            tag_id,
        )
        return label_box_data_rows

    def export_label_box_data_rows_by_tag_name(
        self,
        tag_name: str,
    ) -> List[Dict]:
        """Exports samples in a format compatible with Labelbox.

        The format is documented here:
        https://docs.labelbox.com/docs/images-json

        Args:
            tag_name:
                Name of the tag which should exported.

        Returns:
            A list of dictionaries in a format compatible with Labelbox.

        Examples:
            >>> # write json file which can be imported in Label Studio
            >>> tasks = client.export_label_box_data_rows_by_tag_name(
            >>>     'initial-tag'
            >>> )
            >>> 
            >>> with open('my-labelbox-rows.json', 'w') as f:
            >>>     json.dump(tasks, f)

        """
        tag = self.get_tag_by_name(tag_name)
        return self.export_label_box_data_rows_by_tag_id(tag.id)