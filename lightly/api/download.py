""" Download from Lightly API """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import List

from lightly.api import routes


def get_samples_by_tag(tag_name: str,
                       dataset_id: str,
                       token: str,
                       mode: str = 'list',
                       filenames: List[str] = None):
    """Get the files associated with a given tag and dataset.

    Asks the servers for all samples in a given tag and dataset. If mode is
    mask or indices, the list of all filenames must be specified. Can return
    either the list of all filenames in the tag, or a mask or indices
    indicating which of the provided filenames are in the tag.

    Args:
        tag_name:
            Name of the tag to query.
        dataset_id:
            The unique identifier of the dataset.
        token:
            Token for authentication.
        mode:
            Return type, must be in ["list", "mask", "indices"].
        filenames:
            List of all filenames.

    Returns:
        Either list of filenames, binary mask, or list of indices
        specifying the samples in the requested tag.

    Raises:
        ValueError, AssertionError

    """

    if mode == 'mask' and filenames is None:
        msg = f'Argument filenames must not be None for mode "{mode}"!'
        raise ValueError(msg)
    if mode == 'indices' and filenames is None:
        msg = f'Argument filenames must not be None for mode "{mode}"!'
        raise ValueError(msg)
    if mode not in ['list', 'mask', 'indices']:
        msg = f'Got illegal mode "{mode}"! '
        msg += 'Must be in ["list", "mask", "indices"]'
        raise ValueError(msg)

    samples = routes.users.datasets.tags.get_samples(
        dataset_id, token, tag_name=tag_name)

    if mode == 'list':
        return samples
    elif mode == 'mask':
        mask = [1 if f in set(samples) else 0 for f in filenames]
        assert sum(mask) == len(samples)
        return mask
    else:
        indices = [i for i in range(len(filenames))]
        indices = filter(lambda i: filenames[i] in set(samples), indices)
        indices = list(indices)
        assert len(indices) == len(samples)
        return indices
