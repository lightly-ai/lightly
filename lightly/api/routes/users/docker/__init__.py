""" Docker Routes

This site is under construction.

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import lightly.api.routes.users as users

__module = 'docker'


def _prefix(*args, **kwargs):
    """Returns the prefix for the docker routes.

    """
    return users._prefix() + '/' + __module
