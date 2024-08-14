# -*- coding: utf-8 -*-
"""**Lightly Magic:** Train and embed in one command.

This module contains the entrypoint for the **lightly-magic-deprecated**
command-line interface.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import hydra
from omegaconf import DictConfig

from lightly.cli._helpers import fix_hydra_arguments
from lightly.cli.embed_cli import _embed_cli
from lightly.cli.train_cli import _train_cli
from lightly.utils.hipify import print_as_warning


def _lightly_cli(cfg, is_cli_call=True):
    print_as_warning(
        "The lightly-magic-deprecated command is deprecated since version 1.6 and "
        "will be removed in version 1.7. If you would like to continue using the "
        "command, please create an issue on the issue tracker at "
        "https://github.com/lightly-ai/lightly/issues or contact us at info@lightly.ai"
    )
    cfg["loader"]["shuffle"] = True
    cfg["loader"]["drop_last"] = True

    if cfg["trainer"]["max_epochs"] > 0:
        print("#" * 10 + " Starting to train an embedding model.")
        checkpoint = _train_cli(cfg, is_cli_call)
    else:
        checkpoint = ""

    cfg["loader"]["shuffle"] = False
    cfg["loader"]["drop_last"] = False
    cfg["checkpoint"] = checkpoint

    print("#" * 10 + " Starting to embed your dataset.")
    embeddings = _embed_cli(cfg, is_cli_call)
    cfg["embeddings"] = embeddings

    print("#" * 10 + " Finished")


@hydra.main(**fix_hydra_arguments(config_path="config", config_name="config"))
def lightly_cli(cfg):
    """Train a self-supervised model and use it to embed your dataset.

    .. warning::

        This functionality is deprecated since version 1.6. The lightly-magic
        command was renamed to lightly-magic-deprecated in version 1.6 and will be
        completely removed in version 1.7. If you would like to continue using the
        command, please create an issue on the
        `issue tracker <https://github.com/lightly-ai/lightly/issues>`_
        or contact us at info@lightly.ai

    Args:
        cfg:
            The default configs are loaded from the config file.
            To overwrite them please see the section on the config file
            (.config.config.yaml).

    Command-Line Args:
        input_dir:
            Path to the input directory where images are stored.

    Examples:
        >>> # train model and embed images with default settings
        >>> lightly-magic-deprecated input_dir=data/
        >>>
        >>> # train model for 10 epochs and embed images
        >>> lightly-magic-deprecated input_dir=data/ trainer.max_epochs=10


    """
    return _lightly_cli(cfg)


def entry():
    lightly_cli()
