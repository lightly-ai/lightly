# -*- coding: utf-8 -*-
"""**Lightly Version:** Show the version of the installed package.

Example:
    >>> # show the version of the installed package
    >>> lightly-version
"""

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

import hydra
import lightly

from lightly.cli._helpers import fix_hydra_arguments


def _version_cli():
    version = lightly.__version__
    print(f'lightly version {version}', flush=True)


@hydra.main(**fix_hydra_arguments(config_path = 'config', config_name = 'config'))
def version_cli(cfg):
    """Prints the version of the used lightly package to the terminal.

    """
    _version_cli()


def entry():
    version_cli()
