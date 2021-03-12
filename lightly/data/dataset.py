""" Lightly Dataset """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
import shutil
from PIL import Image
from typing import List, Union

import torch.utils.data as data
import torchvision.datasets as datasets

from lightly.data._helpers import _load_dataset
from lightly.data._helpers import DatasetFolder
from lightly.data._video import VideoDataset


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
    Returns a tuple (sample, target, fname) when accessed using __getitem__

    Can either be used to load a dataset offered by torchvision (e.g. cifar10)
    or to load a custom dataset from an input folder.

    Args:
        input_dir:
            Path to directory holding the images to load.
        transform:
            Image transforms (as in torchvision).
        index_to_filename:
            Function which takes the dataset and index as input and returns
            the filename of the file at the index. If None, uses default.

    Examples:
        >>> # load cifar10 from a local folder
        >>> import lightly.data as data
        >>> dataset = data.LightlyDataset(input_dir='path/to/cifar10/')
        >>> sample, target, fname = dataset[0]

    """

    def __init__(self,
                 input_dir: str,
                 transform=None,
                 index_to_filename=None):

        # can pass input_dir=None to create an "empty" dataset
        self.input_dir = input_dir
        if self.input_dir is not None:
            self.dataset = _load_dataset(self.input_dir, transform)

        # initialize function to get filename of image
        self.index_to_filename = _get_filename_by_index
        if index_to_filename is  not None:
            self.index_to_filename = index_to_filename

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
            transform=transform,
            index_to_filename=index_to_filename
        )

        # populate it with the torch dataset
        dataset_obj.dataset = dataset
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
                Image format.

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
            all_filenames = self.get_filenames()
            for i in range(len(filenames)):
                if filenames[i] in all_filenames:
                    indices.append(i)

        # dump images
        for i, filename in zip(indices, filenames):
            _dump_image(self.dataset, output_dir, filename, i, fmt=format)

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
