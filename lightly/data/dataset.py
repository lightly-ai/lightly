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
    """

    """
    if isinstance(dataset, datasets.ImageFolder):
        full_path = dataset.imgs[index][0]
        return os.path.relpath(full_path, dataset.root)
    elif isinstance(dataset, DatasetFolder):
        full_path = dataset.samples[index][0]
        return os.path.relpath(full_path, dataset.root)
    elif isinstance(dataset, VideoDataset):
        return dataset.get_filename(index)
    else:
        return str(index)


def _ensure_dir(path):
    """

    """
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)


def _copy_image(input_dir, output_dir, filename):
    """

    """
    source = os.path.join(input_dir, filename)
    target = os.path.join(output_dir, filename)
    _ensure_dir(target)
    shutil.copyfile(source, target)

def _save_image(image, output_dir, filename, fmt):
    """

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

    # TODO

    if isinstance(dataset, datasets.ImageFolder):
        _copy_image(dataset.root, output_dir, filename)
    elif isinstance(dataset, DatasetFolder):
        _copy_image(dataset.root, output_dir, filename)
    else:
        image, _ = dataset[index]
        _save_image(image, output_dir, filename, fmt)


class LightlyDataset:
    """Provides a uniform data interface for the embedding models.

    Should be used for all models and functions in the lightly package.
    Returns a tuple (sample, target, fname) when accessed using __getitem__

    Can either be used to load a dataset offered by torchvision (e.g. cifar10)
    or to load a custom dataset from an input folder.

    Args:
        root:
            Directory where the torchvision dataset should be stored.
        name:
            Name of the dataset if it is a torchvision dataset 
            (e.g. cifar10, cifar100).
        train:
            Use the training set if it is a torchvision dataset.
        download:
            Whether to download the torchvision dataset.
        from_folder:
            Path to directory holding the images to load.
        transform:
            Image transforms (as in torchvision).
        indices:
            If provided, ignores samples not in indices.

    Examples:
        >>> import lightly.data as data
        >>> # load cifar10 from torchvision
        >>> dataset = data.LightlyDataset(
        >>>     root='./', name='cifar10', download=True)
        >>> # load cifar10 from a local folder
        >>> dataset = data.LightlyDataset(from_folder='path/to/cifar10/')
        >>> sample, target, fname = dataset[0]

    """

    def __init__(self,
                 input_dir: str,
                 transform=None,
                 index_to_filename=None):

        # TODO
        self.input_dir = input_dir
        if self.input_dir is not None:
            self.dataset = _load_dataset(self.input_dir, transform)

        # TODO
        if index_to_filename is None:
            self.index_to_filename = _get_filename_by_index
        else:
            self.index_to_filename = index_to_filename

    @classmethod
    def from_torch_dataset(cls,
                           dataset,
                           transform=None,
                           index_to_filename=None):
        """

        """
        # TODO
        dataset_obj = cls(
            None,
            transform=transform,
            index_to_filename=index_to_filename
        )
        # TODO
        dataset_obj.dataset = dataset
        return dataset_obj

    def __getitem__(self, index: int):
        """ Get item at index. Supports torchvision.ImageFolder datasets and
            all dataset which return the tuple (sample, target).

        Args:
         - index:   index of the queried item

        Returns:
         - sample:  sample at queried index
         - target:  class_index of target class, 0 if there is no target
         - fname:   filename of the sample, str(index) if there is no filename

        """
        # TODO
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
        """Saves images to the output directory.

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

        # TODO
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


"""
class LightlyDualViewDataset(LightlyDataset):

    def __init__(self,
                 input_dir: str,
                 transform=None,
                 index_to_filename=None):

        super(LightlyDualViewDataset, self).__init__(
            input_dir,
            transform=None,
            index_to_filename=index_to_filename,
        )

        self.transform = transform

    @classmethod
    def from_torch_dataset(cls,
                           dataset,
                           transform=None,
                           index_to_filename=None):

        # TODO
        dataset_obj = cls(
            None,
            transform=None,
            index_to_filename=index_to_filename
        )
        # TODO
        dataset_obj.dataset = dataset
        dataset_obj.transform = transform

        return dataset_obj

    def __getitem__(self, index: int):

        img_1 = self.transform(self.dataset[index])
        img_2 = self.transform(self.dataset[index])
        return img_1, img_2
"""