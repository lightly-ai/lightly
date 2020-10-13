""" Lightly     Dataset """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
from typing import List

import torch.utils.data as data
import torchvision.datasets as datasets

from lightly.data._helpers import _load_dataset
from lightly.data._helpers import DatasetFolder


class LightlyDataset(data.Dataset):
    """Provides a uniform data interface for the embedding models.

    Should be used for all models and functions in the lightly package.

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

    Examples:
        >>> import lightly.data as data
        >>> # load cifar10 from torchvision
        >>> dataset = data.LightlyDataset(
        >>>     root='./', name='cifar10', download=True)
        >>> # load cifar10 from a local folder
        >>> dataset = data.LightlyDataset(from_folder='path/to/cifar10/')

    """

    def __init__(self,
                 root: str = '',
                 name: str = 'cifar10',
                 train: bool = True,
                 download: bool = True,
                 from_folder: str = '',
                 transform=None):
        """ Constructor



        Raises:
            ValueError: If the specified dataset doesn't exist

        """

        super(LightlyDataset, self).__init__()
        self.dataset = _load_dataset(
            root, name, train, download, transform, from_folder
        )
        self.root_folder = None
        if from_folder:
            self.root_folder = from_folder

    def get_filenames(self) -> List[str]:
        """Returns all filenames in the dataset.

        """
        list_of_filenames = []
        for index in range(len(self)):
            fname = self._get_filename_by_index(index)
            list_of_filenames.append(fname)
        return list_of_filenames

    def _get_filename_by_index(self, index) -> str:
        """Returns filename based on index
        """
        if isinstance(self.dataset, datasets.ImageFolder):
            full_path = self.dataset.imgs[index][0]
            return os.path.relpath(full_path, self.root_folder)
        elif isinstance(self.dataset, DatasetFolder):
            full_path = self.dataset.samples[index][0]
            return os.path.relpath(full_path, self.root_folder)
        else:
            return str(index)

    def __getitem__(self, index):
        """ Get item at index. Supports torchvision.ImageFolder datasets and
            all dataset which return the tuple (sample, target).

        Args:
         - index:   index of the queried item

        Returns:
         - sample:  sample at queried index
         - target:  class_index of target class, 0 if there is no target
         - fname:   filename of the sample, str(index) if there is no filename

        """
        fname = self._get_filename_by_index(index)
        sample, target = self.dataset.__getitem__(index)
        return sample, target, fname

    def __len__(self):
        return len(self.dataset)

    def __add__(self, other):
        raise NotImplementedError()
