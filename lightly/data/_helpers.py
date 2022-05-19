""" Helper Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
from typing import List, Set, Optional, Callable, Dict, Any

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


def _dir_contains_videos(root: str, extensions: tuple):
    """Checks whether directory contains video files.

    Args:
        root: Root directory path.

    Returns:
        True if root contains video files.

    """
    with os.scandir(root) as scan_dir:
        return any(f.name.lower().endswith(extensions) for f in scan_dir)


def _contains_videos(root: str, extensions: tuple):
    """Checks whether directory or any subdirectory contains video files.

    Iterates over all subdirectories of "root" recursively and returns True
    if any of the subdirectories contains a file with a VIDEO_EXTENSION.

    Args:
        root: Root directory path.

    Returns:
        True if "root" or any subdir contains video files.

    """
    for subdir, _, _ in os.walk(root):
        if _dir_contains_videos(subdir, extensions):
            return True
    return False


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
        return any(not _is_lightly_output_dir(f.name) for f in scan_dir \
            if f.is_dir())


def _load_dataset_from_folder(
        root: str, transform,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        tqdm_args: Dict[str, Any] = None,
        num_workers_video_frame_counting: int = 0
):
    """Initializes dataset from folder.

    Args:
        root: (str) Root directory path
        transform: (torchvision.transforms.Compose) image transformations

    Returns:
        Dataset consisting of images/videos in the root directory.

    Raises:
        ValueError: If the specified dataset doesn't exist

    """
    if not os.path.exists(root):
        raise ValueError(f'The input directory {root} does not exist!')

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
        dataset = VideoDataset(
            root,
            extensions=VIDEO_EXTENSIONS,
            transform=transform,
            is_valid_file=is_valid_file,
            tqdm_args=tqdm_args,
            num_workers=num_workers_video_frame_counting
        )
    elif _contains_subdirs(root):
        # root contains subdirectories -> create an image folder dataset
        dataset = datasets.ImageFolder(root,
                                       transform=transform,
                                       is_valid_file=is_valid_file
                                       )
    else:
        # root contains plain images -> create a folder dataset
        dataset = DatasetFolder(root,
                                extensions=IMG_EXTENSIONS,
                                transform=transform,
                                is_valid_file=is_valid_file
                                )

    return dataset
