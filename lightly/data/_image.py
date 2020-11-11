""" """

#
#

import os
import torchvision.datasets as datasets


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