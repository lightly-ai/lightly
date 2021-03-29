# -*- coding: utf-8 -*-
"""**Lightly Version:** Show version of installed package.

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import hydra
import lightly

def _version_cli():
    version = lightly.__version__
    print(f'lightly version {version}', flush=True)


@hydra.main(config_path='config', config_name='config')
def version_cli(cfg):
    """Prints the version of the used lightly package to the terminal.

    """
    _version_cli()


def entry():
    version_cli()