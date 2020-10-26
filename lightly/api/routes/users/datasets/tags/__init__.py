""" Tags Routes """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import Union
import lightly.api.routes.users.datasets as datasets

__module = 'tags'


def _prefix(dataset_id: Union[str, None] = None,
            tag_id: Union[str, None] = None,
            *args, **kwargs):
    """Returns the prefix for the tags routes.

    Args:
        dataset_id:
            Identifier of the dataset.
        tag_id:
            Identifier of the tag.

    """
    prefix = datasets._prefix(dataset_id=dataset_id)
    if tag_id is None:
        return prefix + '/' + __module
    else:
        return prefix + '/' + __module + '/' + tag_id


# provided functions
from lightly.api.routes.users.datasets.tags.service import post         # noqa: F401, E402, E501
from lightly.api.routes.users.datasets.tags.service import get          # noqa: F401, E402, E501
from lightly.api.routes.users.datasets.tags.service import get_samples  # noqa: F401, E402, E501
