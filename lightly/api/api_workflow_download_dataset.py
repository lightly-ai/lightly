from typing import Dict, List
import io
import warnings
import os
import tqdm
from urllib.request import Request, urlopen
from PIL import Image

from lightly.api.utils import paginate_endpoint, retry
from torch.utils.hipify.hipify_python import bcolors

from concurrent.futures.thread import ThreadPoolExecutor

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
                         max_workers: int = 8,
                         verbose: bool = True):
        """Downloads images from the web-app and stores them in output_dir.

        Args:
            output_dir:
                Where to store the downloaded images.
            tag_name:
                Name of the tag which should be downloaded.
            max_workers:
                Maximum number of workers downloading images in parallel.
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

        downloadables = zip(sample_ids, filenames)

        # handle the case where len(sample_ids) < max_workers
        max_workers = min(len(sample_ids), max_workers)
        max_workers = max(max_workers, 1)

        if verbose:
            print(f'Downloading {bcolors.OKGREEN}{len(sample_ids)}{bcolors.ENDC} images (with {bcolors.OKGREEN}{max_workers}{bcolors.ENDC} workers):', flush=True)
            pbar = tqdm.tqdm(
                unit='imgs',
                total=len(sample_ids)
            )
            tqdm_lock = tqdm.tqdm.get_lock()

        # define lambda function for concurrent download
        def lambda_(i):
            sample_id, filename = i
            # try to download image
            try:
                read_url = self._samples_api.get_sample_image_read_url_by_id(
                    self.dataset_id, 
                    sample_id,
                    type="full",
                )
                img = _get_image_from_read_url(read_url)
                _make_dir_and_save_image(output_dir, filename, img)
                success = True
            except Exception as e: # pylint: disable=broad-except
                warnings.warn(
                    f'Downloading of image {filename} failed with error {e}'
                )
                success = False

            # update the progress bar
            if verbose:
                tqdm_lock.acquire()
                pbar.update(1)
                tqdm_lock.release()
            # return whether the download was successful
            return success

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda_, downloadables, chunksize=1))

        if not all(results):
            msg = 'Warning: Unsuccessful download! '
            msg += 'Failed at image: {}'.format(results.index(False))
            warnings.warn(msg)





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
        label_studio_tasks = paginate_endpoint(
            self._tags_api.export_tag_to_label_studio_tasks,
            page_size=10000,
            dataset_id=self.dataset_id,
            tag_id=tag_id
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
        label_box_data_rows = paginate_endpoint(
            self._tags_api.export_tag_to_label_box_data_rows,
            page_size=10000,
            dataset_id=self.dataset_id,
            tag_id=tag_id
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


    def export_filenames_by_tag_id(
        self,
        tag_id: str,
    ) -> str:
        """Exports a list of the samples filenames within a certain tag.

        Args:
            tag_id:
                Id of the tag which should exported.

        Returns:
            A list of the samples filenames within a certain tag.

        """
        filenames = retry(
            self._tags_api.export_tag_to_basic_filenames,
            dataset_id=self.dataset_id,
            tag_id=tag_id,
        )
        return filenames

    def export_filenames_by_tag_name(
        self,
        tag_name: str,
    ) -> str:
        """Exports a list of the samples filenames within a certain tag.

        Args:
            tag_name:
                Name of the tag which should exported.

        Returns:
            A list of the samples filenames within a certain tag.

        Examples:
            >>> # write json file which can be imported in Label Studio
            >>> filenames = client.export_filenames_by_tag_name(
            >>>     'initial-tag'
            >>> )
            >>> 
            >>> with open('filenames-of-initial-tag.txt', 'w') as f:
            >>>     f.write(filenames)

        """
        tag = self.get_tag_by_name(tag_name)
        return self.export_filenames_by_tag_id(tag.id)    


    def export_read_url_mapping_by_tag_id(
        self,
        tag_id: str,
    ) -> List[Dict]:
        """Export the samples filenames to map with their readURL.

        Args:
            tag_id:
                Id of the tag which should exported.

        Returns:
            A list of mappings of the samples filenames and readURLs within a certain tag.

        """
        mappings = paginate_endpoint(
            self._tags_api.export_tag_to_basic_read_url_mapping,
            page_size=10000,
            dataset_id=self.dataset_id,
            tag_id=tag_id
        )
        return mappings

    def export_read_url_mapping_by_tag_name(
        self,
        tag_name: str,
    ) -> List[Dict]:
        """Export the samples filenames to map with their readURL.

        Args:
            tag_name:
                Name of the tag which should exported.

        Returns:
            A list of mappings of the samples filenames and readURLs within a certain tag.

        Examples:
            >>> # write json file which can be used to access the actual file contents.
            >>> mappings = client.export_read_url_mapping_by_tag_name(
            >>>     'initial-tag'
            >>> )
            >>> 
            >>> with open('my-readURL-mappings.json', 'w') as f:
            >>>     json.dump(mappings, f)

        """
        tag = self.get_tag_by_name(tag_name)
        return self.export_read_url_mapping_by_tag_id(tag.id)