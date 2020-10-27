""" Communication Routes """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.api.utils import getenv


def _prefix(*args, **kwargs):
    """Returns the lightly server location.

    The server location is the prefix for all api requests.

    """
    return getenv(
        'LIGHTLY_SERVER_LOCATION',
        'https://api.lightly.ai'
    )


# submodules
import lightly.api.routes.pip       # noqa: F401, E402
import lightly.api.routes.users     # noqa: F401, E402
