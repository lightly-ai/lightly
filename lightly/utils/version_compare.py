""" Utility method for comparing versions of libraries """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved


def version_compare(v0: str, v1: str):
    """Returns 1 if version of v0 is larger than v1 and -1 otherwise
    
    Use this method to compare Python package versions and see which one is
    newer.

    Examples:

        >>> # compare two versions
        >>> version_compare('1.2.0', '1.1.2')
        >>> 1
    """
    v0 = [int(n) for n in v0.split('.')][::-1]
    v1 = [int(n) for n in v1.split('.')][::-1]
    assert len(v0) == 3
    assert len(v1) == 3
    pairs = list(zip(v0, v1))[::-1]
    for x, y in pairs:
        if x < y:
            return -1
        if x > y:
            return 1
    return 0
