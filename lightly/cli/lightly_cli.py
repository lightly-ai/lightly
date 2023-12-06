# -*- coding: utf-8 -*-
"""**Lightly Magic:** Train and embed in one command.

This module contains the entrypoint for the **lightly-magic**
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
        >>> lightly-magic input_dir=data/
        >>>
        >>> # train model for 10 epochs and embed images
        >>> lightly-magic input_dir=data/ trainer.max_epochs=10


    """
    return _lightly_cli(cfg)


def entry():
    lightly_cli()
