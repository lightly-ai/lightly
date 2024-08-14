""" The lightly.cli module provides a console interface for training self-supervised
models, embedding, and filtering datasets

.. warning::

    Most commands of the CLI are deprecated since version 1.6.

    The following commands were renamed in version 1.6 and will be removed in version
    1.7:

    - `lightly-crop` -> `lightly-crop-deprecated`
    - `lightly-train` -> `lightly-train-deprecated`
    - `lightly-embed` -> `lightly-embed-deprecated`
    - `lightly-magic` -> `lightly-magic-deprecated`

    If you would like to continue using these commands, please create an issue on the
    `issue tracker <https://github.com/lightly-ai/lightly/issues>`_ or contact us at
    info@lightly.ai
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from lightly.cli.crop_cli import crop_cli
from lightly.cli.download_cli import download_cli
from lightly.cli.embed_cli import embed_cli
from lightly.cli.lightly_cli import lightly_cli
from lightly.cli.train_cli import train_cli
