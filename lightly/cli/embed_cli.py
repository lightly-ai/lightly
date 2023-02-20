# -*- coding: utf-8 -*-
"""**Lightly Embed:** Embed images with one command.

This module contains the entrypoint for the **lightly-embed**
command-line interface.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
from typing import Union, Tuple, List

import hydra
import numpy as np
import torch
import torchvision
from torch.utils.hipify.hipify_python import bcolors

from lightly.data import LightlyDataset

from lightly.cli._helpers import fix_hydra_arguments

from lightly.utils.io import save_embeddings

from lightly.cli._helpers import get_model_from_config
from lightly.cli._helpers import fix_input_path
from lightly.cli._helpers import cpu_count


def _embed_cli(cfg, is_cli_call=True) -> \
    Union[
        Tuple[np.ndarray, List[int], List[str]],
        str
    ]:
    """ See embed_cli() for usage documentation

        is_cli_call:
            If True:
                Saves the embeddings as file and returns the filepath.
            If False:
                Returns the embeddings, labels, filenames as tuple.
                Embeddings are of shape (n_samples, embedding_size)
                len(labels) = len(filenames) = n_samples
    """
    input_dir = cfg['input_dir']
    if input_dir and is_cli_call:
        input_dir = fix_input_path(input_dir)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                (cfg['collate']['input_size'], cfg['collate']['input_size'])
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    dataset = LightlyDataset(input_dir, transform=transform)

    # disable drop_last and shuffle
    cfg['loader']['drop_last'] = False
    cfg['loader']['shuffle'] = False
    cfg['loader']['batch_size'] = min(cfg['loader']['batch_size'], len(dataset))

    # determine the number of available cores
    if cfg['loader']['num_workers'] < 0:
        cfg['loader']['num_workers'] = cpu_count()

    dataloader = torch.utils.data.DataLoader(dataset, **cfg['loader'])

    encoder = get_model_from_config(cfg, is_cli_call)

    embeddings, labels, filenames = encoder.embed(dataloader, device=device)

    if is_cli_call:
        path = os.path.join(os.getcwd(), 'embeddings.csv')
        save_embeddings(path, embeddings, labels, filenames)
        print(f'Embeddings are stored at {bcolors.OKBLUE}{path}{bcolors.ENDC}')
        os.environ[
            cfg['environment_variable_names']['lightly_last_embedding_path']
        ] = path
        return path

    return embeddings, labels, filenames


@hydra.main(**fix_hydra_arguments(config_path = 'config', config_name = 'config'))
def embed_cli(cfg) -> str:
    """Embed images from the command-line.

    Args:
        cfg:
            The default configs are loaded from the config file.
            To overwrite them please see the section on the config file
            (.config.config.yaml).

    Command-Line Args:
        input_dir:
            Path to the input directory where images are stored.
        checkpoint:
            Path to the checkpoint of a pretrained model. If left
            empty, a pretrained model by lightly is used.

    Returns:
        The path to the created embeddings file.

    Examples:
        >>> # embed images with default settings and a lightly model
        >>> lightly-embed input_dir=data/
        >>>
        >>> # embed images with default settings and a custom checkpoint
        >>> lightly-embed input_dir=data/ checkpoint=my_checkpoint.ckpt
        >>>
        >>> # embed images with custom settings
        >>> lightly-embed input_dir=data/ model.num_ftrs=32

    """
    return _embed_cli(cfg)


def entry():
    embed_cli()
