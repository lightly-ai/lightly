""" Helper Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
import torchvision.datasets as datasets

from lightly.data._image_loaders import default_loader

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def _make_dataset(directory, extensions=None, is_valid_file=None):
    """Return a list of all image files with targets in the directory

    Args:
        directory: (str) Root directory path
            (should not contain subdirectories!)
        extensions: (List[str]) List of allowed extensions
        is_valid_file: (callable) Used to check corrupt files

    Returns:
        List of instance tuples: (path_i, target_i = 0)

    """

    if extensions is not None:
        def _is_valid_file(filename):
            return filename.lower().endswith(extensions)

    instances = []
    for fname in os.listdir(directory):

        if not _is_valid_file(fname):
            continue

        path = os.path.join(directory, fname)
        item = (path, 0)
        instances.append(item)

    return instances


class DatasetFolder(datasets.VisionDataset):

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        """Constructor based on torchvisions DatasetFolder
            (https://pytorch.org/docs/stable/torchvision/datasets.html#datasetfolder)

        Args:
            root: (str) Root directory path
            loader: (callable) Function that loads file at path
            extensions: (List[str]) List of allowed extensions
            transform: Function that takes a PIL image and returns
                transformed version
            target_transform: As transform but for targets
            is_valid_file: (callable) Used to check corrupt files

        Raises:
            RuntimeError: If no supported files are found in root.

        """

        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)

        samples = _make_dataset(self.root, extensions, is_valid_file)
        if len(samples) == 0:
            msg = 'Found 0 files in folder: {}\n'.format(self.root)
            if extensions is not None:
                msg += 'Supported extensions are: {}'.format(
                    ','.join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        """Returns item at index

        Args:
            index: (int) Index

        Returns:
            tuple: (sample, target) where target is 0

        """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


def _contains_subdirs(root):
    """Check whether directory contains subdirectories

    Args:
        root: (str) Root directory path

    Returns:
        True if root contains subdirectories else false

    """

    list_dir = os.listdir(root)
    is_dir = [
        os.path.isdir(os.path.join(root, f)) for f in list_dir
    ]
    return any(is_dir)


def _load_dataset_from_folder(root, transform):
    """Initialize dataset from folder

    Args:
        root: (str) Root directory path
        transform: (torchvision.transforms.Compose) image transformations

    Returns:
        Dataset consisting of images in the root directory
    """

    if _contains_subdirs(root):
        dataset = datasets.ImageFolder(root, transform=transform)
    else:
        dataset = DatasetFolder(root, default_loader,
                                extensions=IMG_EXTENSIONS,
                                transform=transform)

    return dataset


def _load_dataset(root='',
                  name='cifar10',
                  train=True,
                  download=True,
                  transform=None,
                  from_folder=''):
    """ Initialize dataset from torchvision or from folder

    Args:
        root: (str) Directory where dataset is stored
        name: (str) Name of the dataset (e.g. cifar10, cifar100)
        train: (bool) Use the training set
        download: (bool) Download the dataset
        transform: (torchvision.transforms.Compose) image transformations
        from_folder: (str) Path to directory holding the images to load.

    Returns:
        A torchvision dataset

    Raises:
        ValueError: If the specified dataset doesn't exist

    """

    if from_folder and os.path.exists(from_folder):
        # load data from directory
        dataset = _load_dataset_from_folder(from_folder,
                                            transform)

    elif name.lower() == 'cifar10' and root:
        # load cifar10
        dataset = datasets.CIFAR10(root,
                                   train=train,
                                   download=download,
                                   transform=transform)

    elif name.lower() == 'cifar100' and root:
        # load cifar100
        dataset = datasets.CIFAR100(root,
                                    train=train,
                                    download=download,
                                    transform=transform)

    elif name.lower() == 'cityscapes' and root:
        # load cityscapes
        root = os.path.join(root, 'cityscapes/')
        split = 'train' if train else 'val'
        dataset = datasets.Cityscapes(root,
                                      split=split,
                                      transform=transform)

    elif name.lower() == 'stl10' and root:
        # load stl10
        split = 'train' if train else 'test'
        dataset = datasets.STL10(root,
                                 split=split,
                                 download=download,
                                 transform=transform)

    elif name.lower() == 'voc07-seg' and root:
        # load pascal voc 07 segmentation dataset
        image_set = 'train' if train else 'val'
        dataset = datasets.VOCSegmentation(root,
                                           year='2007',
                                           image_set=image_set,
                                           download=download,
                                           transform=transform)

    elif name.lower() == 'voc12-seg' and root:
        # load pascal voc 12 segmentation dataset
        image_set = 'train' if train else 'val'
        dataset = datasets.VOCSegmentation(root,
                                           year='2012',
                                           image_set=image_set,
                                           download=download,
                                           transform=transform)

    elif name.lower() == 'voc07-det' and root:
        # load pascal voc 07 object detection dataset
        image_set = 'train' if train else 'val'
        dataset = datasets.VOCDetection(root,
                                        year='2007',
                                        image_set=image_set,
                                        download=True,
                                        transform=transform)

    elif name.lower() == 'voc12-det' and root:
        # load pascal voc 12 object detection dataset
        image_set = 'train' if train else 'val'
        dataset = datasets.VOCDetection(root,
                                        year='2012',
                                        image_set=image_set,
                                        download=True,
                                        transform=transform)

    else:
        raise ValueError(
            'The specified dataset (%s) or datafolder (%s) does not exist '
            % (name, from_folder))

    return dataset
