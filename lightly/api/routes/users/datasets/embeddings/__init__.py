""" Embeddings Routes """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import Union
import lightly.api.routes.users.datasets as datasets

__module = 'embeddings'


def _prefix(dataset_id: Union[str, None] = None,
            *args, **kwargs):
    """Returns the prefix for the embeddings routes.

    Args:
        dataset_id:
            Identifier of the dataset.

    """
    return datasets._prefix(dataset_id=dataset_id) + '/' + __module


# provided functions
from lightly.api.routes.users.datasets.embeddings.service import post           # noqa: F401, E402, E501
from lightly.api.routes.users.datasets.embeddings.service import get_summaries  # noqa: F401, E402, E501
