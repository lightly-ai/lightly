""" Lightly Dataset """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
import bisect
import shutil
import tempfile

from PIL import Image
from typing import List, Union, Callable
from torch._C import Value

import torchvision.datasets as datasets
from torchvision import transforms

from lightly.data._helpers import _load_dataset_from_folder
from lightly.data._helpers import DatasetFolder
from lightly.data._video import VideoDataset
from lightly.utils.io import check_filenames


def _get_filename_by_index(dataset, index):
    """Default function which maps the index of an image to a filename.

    """
    if isinstance(dataset, datasets.ImageFolder):
        # filename is the path of the image relative to the dataset root
        full_path = dataset.imgs[index][0]
        return os.path.relpath(full_path, dataset.root)
    elif isinstance(dataset, DatasetFolder):
        # filename is the path of the image relative to the dataset root
        full_path = dataset.samples[index][0]
        return os.path.relpath(full_path, dataset.root)
    elif isinstance(dataset, VideoDataset):
        # filename is constructed by the video dataset
        return dataset.get_filename(index)
    else:
        # dummy to prevent crashes
        return str(index)


def _ensure_dir(path):
    """Makes sure that the directory at path exists.

    """
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)


def _copy_image(input_dir, output_dir, filename):
    """Copies an image from the input directory to the output directory.

    """
    source = os.path.join(input_dir, filename)
    target = os.path.join(output_dir, filename)
    _ensure_dir(target)
    shutil.copyfile(source, target)


def _save_image(image, output_dir, filename, fmt):
    """Saves an image in the output directory.

    """
    target = os.path.join(output_dir, filename)
    _ensure_dir(target)
    try:
        # try to save the image with the specified format or
        # derive the format from the filename (if format=None)
        image.save(target, format=fmt)
    except ValueError:
        # could not determine format from filename
        image.save(target, format='png')


def _dump_image(dataset, output_dir, filename, index, fmt):
    """Saves a single image to the output directory.

    Will copy the image from the input directory to the output directory
    if possible. If not (e.g. for VideoDatasets), will load the image and
    then save it to the output directory with the specified format.

    """

    if isinstance(dataset, datasets.ImageFolder):
        # can safely copy the image from the input to the output directory
        _copy_image(dataset.root, output_dir, filename)
    elif isinstance(dataset, DatasetFolder):
        # can safely copy the image from the input to the output directory
        _copy_image(dataset.root, output_dir, filename)
    else:
        # need to load the image and save it to the output directory
        image, _ = dataset[index]
        _save_image(image, output_dir, filename, fmt)


class LightlyDataset:
    """Provides a uniform data interface for the embedding models.

    Should be used for all models and functions in the lightly package.
    Returns a tuple (sample, target, fname) when accessed using __getitem__.

    The LightlyDataset supports different input sources. You can use it
    on a folder of images. You can also use it on a folder with subfolders
    with images (ImageNet style). If the input_dir has subfolders,
    each subfolder gets its own target label.
    You can also work with videos (requires pyav).
    If there are multiple videos in the input_dir each video gets a different
    target label assigned. If input_dir contains images and videos
    only the videos are used.

    Can also be used in combination with the `from_torch_dataset` method
    to load a dataset offered by torchvision (e.g. cifar10).

    Parameters:
        input_dir:
            Path to directory holding the images or videos to load.
        transform:
            Image transforms (as in torchvision).
        index_to_filename:
            Function which takes the dataset and index as input and returns
            the filename of the file at the index. If None, uses default.
        filenames:
            If not None, it filters the dataset in the input directory
            by the given filenames.

    Examples:
        >>> # load a dataset consisting of images from a local folder
        >>> # mydata/
        >>> # `- img1.png
        >>> # `- img2.png
        >>> # `- ...
        >>> import lightly.data as data
        >>> dataset = data.LightlyDataset(input_dir='path/to/mydata/')
        >>> sample, target, fname = dataset[0]
        >>>
        >>> # also works with subfolders
        >>> # mydata/
        >>> # `- subfolder1
        >>> #     `- img1.png
        >>> # `- subfolder2
        >>> # ...
        >>>
        >>> # also works with videos
        >>> # mydata/
        >>> # `- video1.mp4
        >>> # `- video2.mp4
        >>> # `- ...
    """

    def __init__(self,
                 input_dir: Union[str, None],
                 transform: transforms.Compose = None,
                 index_to_filename:
                 Callable[[datasets.VisionDataset, int], str] = None,
                 filenames: List[str] = None):

        # can pass input_dir=None to create an "empty" dataset
        self.input_dir = input_dir
        if filenames is not None:
            filepaths = [
                os.path.join(input_dir, filename)
                for filename in filenames
            ]
            filepaths = set(filepaths)

            def is_valid_file(filepath: str):
                return filepath in filepaths
        else:
            is_valid_file = None

        if self.input_dir is not None:
            self.dataset = _load_dataset_from_folder(
                self.input_dir, transform, is_valid_file=is_valid_file
            )
        elif transform is not None:
            raise ValueError(
                'transform must be None when input_dir is None but is '
                f'{transform}',
            )

        # initialize function to get filename of image
        self.index_to_filename = _get_filename_by_index
        if index_to_filename is not None:
            self.index_to_filename = index_to_filename

        # if created from an input directory with filenames, check if they
        # are valid
        if input_dir:
            check_filenames(self.get_filenames())

    @classmethod
    def from_torch_dataset(cls,
                           dataset,
                           transform=None,
                           index_to_filename=None):
        """Builds a LightlyDataset from a PyTorch (or torchvision) dataset.

        Args:
            dataset:
                PyTorch/torchvision dataset.
            transform:
                Image transforms (as in torchvision).
            index_to_filename:
                Function which takes the dataset and index as input and returns
                the filename of the file at the index. If None, uses default.

        Returns:
            A LightlyDataset object.

        Examples:
            >>> # load cifar10 from torchvision
            >>> import torchvision
            >>> import lightly.data as data
            >>> base = torchvision.datasets.CIFAR10(root='./')
            >>> dataset = data.LightlyDataset.from_torch_dataset(base)

        """
        # create an "empty" dataset object
        dataset_obj = cls(
            None,
            index_to_filename=index_to_filename,
        )

        # populate it with the torch dataset
        dataset_obj.dataset = dataset
        dataset_obj.transform = transform
        return dataset_obj

    def __getitem__(self, index: int):
        """Returns (sample, target, fname) of item at index.

        Args:
            index:
                Index of the queried item.

        Returns:
            The image, target, and filename of the item at index.

        """
        fname = self.index_to_filename(self.dataset, index)
        sample, target = self.dataset.__getitem__(index)

        return sample, target, fname

    def __len__(self):
        """Returns the length of the dataset.

        """
        return len(self.dataset)

    def __add__(self, other):
        """Adds another item to the dataset.

        """
        raise NotImplementedError()

    def get_filenames(self) -> List[str]:
        """Returns all filenames in the dataset.

        """
        if hasattr(self.dataset, 'get_filenames'):
            return self.dataset.get_filenames()

        list_of_filenames = []
        for index in range(len(self)):
            fname = self.index_to_filename(self.dataset, index)
            list_of_filenames.append(fname)
        return list_of_filenames

    def dump(self,
             output_dir: str,
             filenames: Union[List[str], None] = None,
             format: Union[str, None] = None):
        """Saves images in the dataset to the output directory.

        Will copy the images from the input directory to the output directory
        if possible. If not (e.g. for VideoDatasets), will load the images and
        then save them to the output directory with the specified format.

        Args:
            output_dir:
                Output directory where the image is stored.
            filenames:
                Filenames of the images to store. If None, stores all images.
            format:
                Image format. Can be any pillow image format (png, jpg, ...).
                By default we try to use the same format as the input data. If
                not possible (e.g. for videos) we dump the image 
                as a png image to prevent compression artifacts.

        """

        if self.dataset.transform is not None:
            raise RuntimeError('Cannot dump dataset which applies transforms!')

        # create directory if it doesn't exist yet
        os.makedirs(output_dir, exist_ok=True)

        # dump all the files if no filenames were passed, otherwise dump only
        # the ones referenced in the list
        if filenames is None:
            indices = [i for i in range(self.__len__())]
            filenames = self.get_filenames()
        else:
            indices = []
            filenames = sorted(filenames)
            all_filenames = self.get_filenames()
            for index, filename in enumerate(all_filenames):
                filename_index = bisect.bisect_left(filenames, filename)
                # make sure the filename exists in filenames
                if filename_index < len(filenames) and \
                        filenames[filename_index] == filename:
                    indices.append(index)

        # dump images
        for i, filename in zip(indices, filenames):
            _dump_image(self.dataset, output_dir, filename, i, fmt=format)

    def get_filepath_from_filename(self, filename: str, image: Image = None):
        """Returns the filepath given the filename of the image

        There are three cases:
            - The dataset is a regular dataset with the images in the input dir.
            - The dataset is a video dataset, thus the images have to be saved in a
              temporary folder.
            - The dataset is a torch dataset, thus the images have to be saved in a
              temporary folder.

        Args:
            filename:
                The filename of the image
            image:
                The image corresponding to the filename

        Returns:
            The filename to the image, either the existing one (case 1) or a
            newly created jpg (case 2, 3)

        """

        has_input_dir = hasattr(self, 'input_dir') and \
            isinstance(self.input_dir, str)
        if has_input_dir:
            path_to_image = os.path.join(self.input_dir, filename)
            if os.path.isfile(path_to_image):
                # the file exists, return its filepath
                return path_to_image

        if image is None:
            raise ValueError(
                'The parameter image must not be None for'
                'VideoDatasets and TorchDatasets'
            )

        # the file doesn't exist, save it as a jpg and return filepath
        folder_path = tempfile.mkdtemp()
        filepath = os.path.join(folder_path, filename) + '.jpg'
        
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        image.save(filepath)
        return filepath

    @property
    def transform(self):
        """Getter for the transform of the dataset.

        """
        return self.dataset.transform

    @transform.setter
    def transform(self, t):
        """Setter for the transform of the dataset.

        """
        self.dataset.transform = t
