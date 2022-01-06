""" Image Dataset """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import List, Tuple, Set

import os
import torchvision.datasets as datasets
from torchvision import transforms

from lightly.data._image_loaders import default_loader


def _make_dataset(directory, extensions=None, is_valid_file=None) -> List[Tuple[str, int]]:
    """Returns a list of all image files with targets in the directory.

    Args:
        directory:
            Root directory path (should not contain subdirectories!).
        extensions:
            Tuple of valid extensions.
        is_valid_file:
            Used to find valid files.

    Returns:
        List of instance tuples: (path_i, target_i = 0).

    """

    if extensions is None:
        if is_valid_file is None:
            ValueError('Both extensions and is_valid_file cannot be None')
        else:
            _is_valid_file = is_valid_file
    else:
        def is_valid_file_extension(filepath):
            return filepath.lower().endswith(extensions)
        if is_valid_file is None:
            _is_valid_file = is_valid_file_extension
        else:
            def _is_valid_file(filepath):
                return is_valid_file_extension(filepath) and is_valid_file(filepath)

    instances = []
    for f in os.scandir(directory):

        if not _is_valid_file(f.path):
            continue

        # convention: the label of all images is 0, based on the fact that
        # they are all in the same directory
        item = (f.path, 0)
        instances.append(item)

    return sorted(instances, key=lambda x: x[0]) # sort by path


class DatasetFolder(datasets.VisionDataset):
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

    def __init__(self,
                 root: str,
                 loader=default_loader,
                 extensions=None,
                 transform=None,
                 target_transform=None,
                 is_valid_file=None,
                 ):

        super(DatasetFolder, self).__init__(root,
                                            transform=transform,
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

    def __getitem__(self, index: int):
        """Returns item at index.

        Args:
            index:
                Index of the sample to retrieve.

        Returns:
            A tuple (sample, target) where target is 0.

        """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        """Returns the number of samples in the dataset.

        """
        return len(self.samples)
