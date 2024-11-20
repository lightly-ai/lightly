""" Image Dataset """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import torch
import torchvision.datasets as datasets
from typing_extensions import Protocol

from lightly.data._image_loaders import default_loader


class DatasetFolder(datasets.VisionDataset):  # type: ignore
    """Implements a dataset folder.

    DatasetFolder based on torchvisions implementation.
    (https://pytorch.org/docs/stable/torchvision/datasets.html#datasetfolder)

    Attributes:
        root:
            Root directory path
        loader:
            Function that loads file at path
        extensions:
            Tuple of allowed extensions
        transform:
            Function that takes a PIL image and returns transformed version
        target_transform:
            As transform but for targets
        is_valid_file:
            Used to check corrupt files

    Raises:
        RuntimeError: If no supported files are found in root.

    """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any] = default_loader,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        """Initialize a DatasetFolder dataset.

        Args:
            root:
                Path to the root directory containing image files.
            loader:
                A function to load an image from a file path. Defaults to default_loader.
            extensions:
                A tuple of allowed file extensions. If None, is_valid_file must be provided.
            transform:
                Optional transform to be applied to the input image.
            target_transform:
                Optional transform to be applied to the target.
            is_valid_file:
                Optional function to validate file paths. If None and extensions is None,
                raises a ValueError.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        samples = _make_dataset(self.root, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in folder: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Retrieve a sample from the dataset.

        Args:
            index:
                Index of the sample to retrieve.

        Returns:
            A tuple containing the image sample and its target (always 0 in this implementation).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            Total count of samples in the dataset.
        """
        return len(self.samples)


def _make_dataset(
    directory: str,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Create a list of valid image files in the given directory.

    Args:
        directory:
            Root directory path containing image files (should not contain subdirectories).
        extensions:
            Tuple of valid file extensions. If None, is_valid_file must be used.
        is_valid_file:
            Optional function to validate file paths beyond extension checking.

    Returns:
        A list of tuples, where each tuple contains:
        - Full path to an image file
        - Target label (always 0 in this implementation)

    Raises:
        ValueError: If both extensions and is_valid_file are None.
    """
    if extensions is None:
        if is_valid_file is None:
            raise ValueError("Both extensions and is_valid_file cannot be None")
        _is_valid_file = is_valid_file
    else:

        def is_valid_file_extension(filepath: str) -> bool:
            return filepath.lower().endswith(extensions)

        if is_valid_file is None:
            _is_valid_file = is_valid_file_extension
        else:

            def _is_valid_file(filepath: str) -> bool:
                return is_valid_file_extension(filepath) and is_valid_file(filepath)

    instances: List[Tuple[str, int]] = []
    for f in os.scandir(directory):
        if not _is_valid_file(f.path):
            continue

        # convention: the label of all images is 0, based on the fact that
        # they are all in the same directory
        item = (f.path, 0)
        instances.append(item)

    return sorted(instances, key=lambda x: x[0])  # sort by path
