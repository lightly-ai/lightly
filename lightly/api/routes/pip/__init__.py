""" PIP Package Route """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import lightly.api.routes as routes

__module = 'pip'


def _prefix(*args, **kwargs):
    """Returns the prefix for the pip route.

    The pip route does not require authentication.

    """
    return routes._prefix(*args, **kwargs) + '/' + __module


# provided functions
from lightly.api.routes.pip.service import get_version  # noqa: F401, E402
