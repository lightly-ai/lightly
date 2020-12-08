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

class LightlyDataset(data.Dataset):
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
            TODO

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
                 root: str = '',
                 name: str = 'cifar10',
                 train: bool = True,
                 download: bool = True,
                 from_folder: str = '',
                 transform=None,
                 indices=None):

        super(LightlyDataset, self).__init__()
        self.dataset = _load_dataset(
            root, name, train, download, transform, from_folder
        )
        self.root_folder = None
        if from_folder:
            self.root_folder = from_folder
        
        self.indices = indices

    def dump_image(self,
                   output_dir: str,
                   index: int,
                   format: Union[str, None] = None):
        """Saves a single image to the output directory.

        Will copy the image from the input directory to the output directory
        if possible. If not (e.g. for VideoDatasets), will load the image and
        then save it to the output directory with the specified format.

        Args:
            output_dir:
                Output directory where the image is stored.
            index:
                Index of the image to store.
            format:
                Image format.

        """
        if self.indices is not None:
            index = self.indices[index]

        image, _ = self.dataset[index]
        filename = self._get_filename_by_index(index)

        source = os.path.join(self.root_folder, filename)
        target = os.path.join(output_dir, filename)

        dirname = os.path.dirname(target)
        os.makedirs(dirname, exist_ok=True)

        if os.path.isfile(source):
            # copy the file from the source to the target
            shutil.copyfile(source, target)
        else:
            # the source is not a file (e.g. when loading a video frame)
            try:
                # try to save the image with the specified format or
                # derive the format from the filename (if format=None)
                image.save(target, format=format)
            except ValueError:
                # could not determine format from filename
                image.save(os.path.join(output_dir, filename), format='png')

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
        # make sure no transforms are applied to the images
        if self.dataset.transform is not None:
            pass

        # create directory if it doesn't exist yet
        os.makedirs(output_dir, exist_ok=True)

        # get all filenames
        if filenames is None:
            indices = [i for i in range(self.__len__())]
        else:
            indices = \
                [i for i, f in enumerate(self.get_filenames()) if f in filenames]

        # dump images
        for index in indices:
            self.dump_image(output_dir, index, format=format)

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
        if self.indices is not None:
            index = self.indices[index]

        if isinstance(self.dataset, datasets.ImageFolder):
            full_path = self.dataset.imgs[index][0]
            return os.path.relpath(full_path, self.root_folder)
        elif isinstance(self.dataset, DatasetFolder):
            full_path = self.dataset.samples[index][0]
            return os.path.relpath(full_path, self.root_folder)
        elif isinstance(self.dataset, VideoDataset):
            return self.dataset.get_filename(index)
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
        
        if self.indices is not None:
            index = self.indices[index]
        
        sample, target = self.dataset.__getitem__(index)
        
        return sample, target, fname

    def __len__(self):
        """TODO

        """
        if self.indices is not None:
            return len(self.indices)

        return len(self.dataset)

    def __add__(self, other):
        raise NotImplementedError()
