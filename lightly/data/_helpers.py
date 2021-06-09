""" Helper Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
from torchvision import datasets

from lightly.data._image import DatasetFolder

try:
    from lightly.data._video import VideoDataset
    VIDEO_DATASET_AVAILABLE = True
except Exception as e:
    VIDEO_DATASET_AVAILABLE = False
    VIDEO_DATASET_ERRORMSG = e


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mpg',
                    '.hevc', '.m4v', '.webm', '.mpeg')


def _contains_videos(root: str, extensions: tuple):
    """Checks whether directory contains video files.

    Args:
        root: Root directory path.

    Returns:
        True if root contains subdirectories else false.
    """
    with os.scandir(root) as scan_dir:
        return any(f.name.lower().endswith(extensions) for f in scan_dir)


def _is_lightly_output_dir(dirname: str):
    """Checks whether the directory is a lightly_output directory.

    Args:
        dirname: Directory to check.

    Returns:
        True if dirname is "lightly_outputs" else false.

    """
    return 'lightly_outputs' in dirname


def _contains_subdirs(root: str):
    """Checks whether directory contains subdirectories.

    Args:
        root: Root directory path.

    Returns:
        True if root contains subdirectories else false.

    """
    with os.scandir(root) as scan_dir:
        return any(f.is_dir() for f in scan_dir \
            if not _is_lightly_output_dir(f.name))


def _load_dataset_from_folder(root: str, transform):
    """Initializes dataset from folder.

    Args:
        root: (str) Root directory path
        transform: (torchvision.transforms.Compose) image transformations

    Returns:
        Dataset consisting of images in the root directory.

    """

    # if there is a video in the input directory but we do not have
    # the right dependencies, raise a ValueError
    contains_videos = _contains_videos(root, VIDEO_EXTENSIONS)
    if contains_videos and not VIDEO_DATASET_AVAILABLE:
        raise ValueError(f'The input directory {root} contains videos '
                         'but the VideoDataset is not available. \n'
                         'Make sure you have installed the right '
                         'dependencies. The error from the imported '
                         f'module was: {VIDEO_DATASET_ERRORMSG}')

    if contains_videos:
        # root contains videos -> create a video dataset
        dataset = VideoDataset(root,
                               extensions=VIDEO_EXTENSIONS,
                               transform=transform)
    elif _contains_subdirs(root):
        # root contains subdirectories -> create an image folder dataset
        dataset = datasets.ImageFolder(root,
                                       transform=transform)
    else:
        # root contains plain images -> create a folder dataset
        dataset = DatasetFolder(root,
                                extensions=IMG_EXTENSIONS,
                                transform=transform)

    return dataset


def _load_dataset(input_dir: str,
                  transform=None):
    """Initializes dataset from torchvision or from folder.

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

    if not os.path.exists(input_dir):
        raise ValueError(f'The input directory {input_dir} does not exist!')

    return _load_dataset_from_folder(input_dir, transform)
