"""Provides functionality to identify corrupt images in a directory.

This module helps users identify corrupt or unreadable image files within a specified
directory. It uses parallel processing to efficiently scan through large collections
of images.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
from __future__ import annotations

import os

import tqdm.contrib.concurrent as concurrent
from PIL import Image, UnidentifiedImageError

from lightly.data import LightlyDataset


def check_images(data_dir: str) -> tuple[list[str], list[str]]:
    """Identifies corrupt and healthy images in the specified directory.

    The function attempts to open each image file in the directory to verify
    its integrity. It processes images in parallel for better performance.

    Args:
        data_dir: Directory path containing the image files to check.

    Returns:
        A tuple containing two lists:
            - List of filenames of healthy images that can be opened successfully
            - List of filenames of corrupt images that cannot be opened

    Example:
        >>> healthy, corrupt = check_images("path/to/images")
        >>> print(f"Found {len(corrupt)} corrupt images")
    """
    dataset = LightlyDataset(input_dir=data_dir)
    filenames = dataset.get_filenames()

    def _is_corrupt(filename: str) -> bool:
        """Checks if a single image file is corrupt.

        Args:
            filename: Name of the image file to check.

        Returns:
            True if the image is corrupt, False otherwise.
        """
        try:
            image = Image.open(os.path.join(data_dir, filename))
            image.load()
        except (IOError, UnidentifiedImageError):
            return True
        else:
            return False

    mapped = concurrent.thread_map(
        _is_corrupt, filenames, chunksize=min(32, len(filenames))
    )
    healthy_images = [f for f, is_corrupt in zip(filenames, mapped) if not is_corrupt]
    corrupt_images = [f for f, is_corrupt in zip(filenames, mapped) if is_corrupt]
    return healthy_images, corrupt_images
