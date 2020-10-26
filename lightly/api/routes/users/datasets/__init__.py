""" Datasets Routes """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import Union
import lightly.api.routes.users as users

__module = 'datasets'


def _prefix(dataset_id: Union[str, None] = None, *args, **kwargs):
    """Returns the prefix for the datasets routes.

    Args:
        dataset_id:
            The identifier of the dataset.

    """
    prefix = users._prefix(*args, **kwargs) + '/' + __module
    if dataset_id is None:
        return prefix
    else:
        return prefix + '/' + dataset_id


# provided functions
from lightly.api.routes.users.datasets.service import put_image_type  # noqa: F401, E402, E501

# submodules
import lightly.api.routes.users.datasets.embeddings  # noqa: F401, E402
import lightly.api.routes.users.datasets.samples     # noqa: F401, E402
import lightly.api.routes.users.datasets.tags        # noqa: F401, E402
