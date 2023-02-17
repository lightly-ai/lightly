# -*- coding: utf-8 -*-
"""**Lightly Magic:** Train, embed, and upload in one command.

This module contains the entrypoint for the **lightly-magic**
command-line interface.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import hydra
from omegaconf import DictConfig

from lightly.utils.hipify import print_as_warning
from lightly.cli._helpers import fix_hydra_arguments
from lightly.cli.train_cli import _train_cli
from lightly.cli.embed_cli import _embed_cli
from lightly.cli.upload_cli import _upload_cli


def validate_cfg(cfg: DictConfig) -> bool:
    """ Validates a config

    Prints warnings if it is not.
    Args:
        cfg:
            The hydra config object

    Returns:
        Wether the config is valid

    """
    valid = True
    if cfg['trainer']['max_epochs'] > 0 and cfg['append']:
        print_as_warning('When appending to an existing dataset you must '
                         'use the same embedding model. Thus specify '
                         'trainer.max_epochs=0. If you had trained your own model, '
                         'you can use it with checkpoint="path/to/model.ckp".')
        valid = False
    return valid


def _lightly_cli(cfg, is_cli_call=True):
    cfg['loader']['shuffle'] = True
    cfg['loader']['drop_last'] = True

    if not validate_cfg(cfg):
        return

    if cfg['trainer']['max_epochs'] > 0:
        print('#' * 10 + ' Starting to train an embedding model.')
        checkpoint = _train_cli(cfg, is_cli_call)
    else:
        checkpoint = ''

    cfg['loader']['shuffle'] = False
    cfg['loader']['drop_last'] = False
    cfg['checkpoint'] = checkpoint

    print('#' * 10 + ' Starting to embed your dataset.')
    embeddings = _embed_cli(cfg, is_cli_call)
    cfg['embeddings'] = embeddings

    if cfg['token'] and (cfg['dataset_id'] or cfg['new_dataset_name']):
        print('#' * 10 + ' Starting to upload your dataset to the Lightly platform.')
        _upload_cli(cfg)

    print('#' * 10 + ' Finished')


@hydra.main(**fix_hydra_arguments(config_path = 'config', config_name = 'config'))
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
        token:
            User access token to the Lightly platform. If dataset_id
            and token are specified, the images and embeddings are 
            uploaded to the platform.
            (Required for upload)
        dataset_id:
            Identifier of the dataset on the Lightly platform. If 
            dataset_id and token are specified, the images and 
            embeddings are uploaded to the platform.
            (Required for upload)
        custom_metadata:
            Path to a .json file containing custom metadata. The file must be in
            the COCO annotations (although annotations can be empty) format and
            contain an additional field `metadata` storing a list of metadata entries.
            The metadata entries are matched with the images via `image_id`.

    Examples:
        >>> # train model and embed images with default settings
        >>> lightly-magic input_dir=data/
        >>>
        >>> # train model for 10 epochs and embed images
        >>> lightly-magic input_dir=data/ trainer.max_epochs=10
        >>>
        >>> # train model, embed images, and upload to the Lightly platform
        >>> lightly-magic input_dir=data/ token='123' dataset_id='XYZ'
        >>>
        >>> # upload images, embeddings, and custom metadata
        >>> lightly-magic input_dir=data/ token='123' dataset_id='XYZ' custom_metadata=custom_metadata.json


    """
    return _lightly_cli(cfg)


def entry():
    lightly_cli()
