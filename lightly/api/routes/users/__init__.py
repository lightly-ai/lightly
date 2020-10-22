""" Users Routes """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import lightly.api.routes as routes

__module = 'users'


def _prefix(*args, **kwargs):
    """Returns the prefix for the users routes.

    All routes through users require authentication via jwt.

    """
    return routes._prefix() + '/' + __module


# provided functions
from lightly.api.routes.users.service import get_quota  # noqa: F401, E402

# submodules
import lightly.api.routes.users.datasets    # noqa: F401, E402
import lightly.api.routes.users.docker      # noqa: F401, E402
