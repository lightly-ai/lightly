""" Samples Routes """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import Union
import lightly.api.routes.users.datasets as datasets

__module = 'samples'


def _prefix(dataset_id: Union[str, None] = None,
            sample_id: Union[str, None] = None,
            *args, **kwargs):
    """Returns the prefix for the samples routes.

    Args:
        dataset_id:
            Identifier of the dataset.
        sample_id:
            Identifier of the sample.

    """
    prefix = datasets._prefix(dataset_id=dataset_id)
    if sample_id is None:
        return prefix + '/' + __module
    else:
        return prefix + '/' + __module + '/' + sample_id


# provided functions
from lightly.api.routes.users.datasets.samples.service import get_presigned_upload_url  # noqa: F401, E402, E501
from lightly.api.routes.users.datasets.samples.service import post                      # noqa: F401, E402, E501
