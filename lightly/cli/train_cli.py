# -*- coding: utf-8 -*-
"""**Lightly Train:** Train a self-supervised model from the command-line.

This module contains the entrypoint for the **lightly-train**
command-line interface.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
import os

import hydra
import torch
import torch.nn as nn
import warnings

from torch.utils.hipify.hipify_python import bcolors

from lightly.cli._cli_simclr import _SimCLR
from lightly.data import ImageCollateFunction
from lightly.data import LightlyDataset
from lightly.embedding import SelfSupervisedEmbedding
from lightly.loss import NTXentLoss

from lightly.models import ResNetGenerator
from lightly.models.batchnorm import get_norm_layer

from lightly.cli._helpers import is_url
from lightly.cli._helpers import get_ptmodel_from_config
from lightly.cli._helpers import fix_input_path
from lightly.cli._helpers import load_state_dict_from_url
from lightly.cli._helpers import load_from_state_dict
from lightly.cli._helpers import cpu_count
from lightly.cli._helpers import fix_hydra_arguments


def _train_cli(cfg, is_cli_call=True):

    input_dir = cfg['input_dir']
    if input_dir and is_cli_call:
        input_dir = fix_input_path(input_dir)

    if 'seed' in cfg.keys():
        seed = cfg['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if cfg["trainer"]["weights_summary"] == "None":
        cfg["trainer"]["weights_summary"] = None

    if torch.cuda.is_available():
        device = 'cuda'
    elif cfg['trainer'] and cfg['trainer']['gpus']:
        device = 'cpu'
        cfg['trainer']['gpus'] = 0
    else:
        device = 'cpu'
    
    distributed_strategy = None
    if cfg['trainer']['gpus'] > 1:
        distributed_strategy = 'ddp'

    if cfg['loader']['batch_size'] < 64:
        msg = 'Training a self-supervised model with a small batch size: {}! '
        msg = msg.format(cfg['loader']['batch_size'])
        msg += 'Small batch size may harm embedding quality. '
        msg += 'You can specify the batch size via the loader key-word: '
        msg += 'loader.batch_size=BSZ'
        warnings.warn(msg)

    # determine the number of available cores
    if cfg['loader']['num_workers'] < 0:
        cfg['loader']['num_workers'] = cpu_count()

    state_dict = None
    checkpoint = cfg['checkpoint']
    if cfg['pre_trained'] and not checkpoint:
        # if checkpoint wasn't specified explicitly and pre_trained is True
        # try to load the checkpoint from the model zoo
        checkpoint, key = get_ptmodel_from_config(cfg['model'])
        if not checkpoint:
            msg = 'Cannot download checkpoint for key {} '.format(key)
            msg += 'because it does not exist! '
            msg += 'Model will be trained from scratch.'
            warnings.warn(msg)
    elif checkpoint:
        checkpoint = fix_input_path(checkpoint) if is_cli_call else checkpoint
    
    if checkpoint:
        # load the PyTorch state dictionary and map it to the current device
        if is_url(checkpoint):
            state_dict = load_state_dict_from_url(
                checkpoint, map_location=device
            )['state_dict']
        else:
            state_dict = torch.load(
                checkpoint, map_location=device
            )['state_dict']

    # load model
    resnet = ResNetGenerator(cfg['model']['name'], cfg['model']['width'])
    last_conv_channels = list(resnet.children())[-1].in_features
    features = nn.Sequential(
        get_norm_layer(3, 0),
        *list(resnet.children())[:-1],
        nn.Conv2d(last_conv_channels, cfg['model']['num_ftrs'], 1),
        nn.AdaptiveAvgPool2d(1),
    )

    model = _SimCLR(
        features,
        num_ftrs=cfg['model']['num_ftrs'],
        out_dim=cfg['model']['out_dim']
    )
    if state_dict is not None:
        load_from_state_dict(model, state_dict)

    criterion = NTXentLoss(**cfg['criterion'])
    optimizer = torch.optim.SGD(model.parameters(), **cfg['optimizer'])

    dataset = LightlyDataset(input_dir)

    cfg['loader']['batch_size'] = min(
        cfg['loader']['batch_size'],
        len(dataset)
    )

    collate_fn = ImageCollateFunction(**cfg['collate'])
    dataloader = torch.utils.data.DataLoader(dataset,
                                             **cfg['loader'],
                                             collate_fn=collate_fn)

    encoder = SelfSupervisedEmbedding(model, criterion, optimizer, dataloader)
    encoder.init_checkpoint_callback(**cfg['checkpoint_callback'])
    encoder.train_embedding(**cfg['trainer'], strategy=distributed_strategy)

    print(f'Best model is stored at: {bcolors.OKBLUE}{encoder.checkpoint}{bcolors.ENDC}')
    os.environ[
        cfg['environment_variable_names']['lightly_last_checkpoint_path']
    ] = encoder.checkpoint
    return encoder.checkpoint


@hydra.main(**fix_hydra_arguments(config_path = 'config', config_name = 'config'))
def train_cli(cfg):
    """Train a self-supervised model from the command-line.

    Args:
        cfg:
            The default configs are loaded from the config file.
            To overwrite them please see the section on the config file 
            (.config.config.yaml).
    
    Command-Line Args:
        input_dir:
            Path to the input directory where images are stored.

    Examples:
        >>> # train model with default settings
        >>> lightly-train input_dir=data/
        >>>
        >>> # train model with batches of size 128
        >>> lightly-train input_dir=data/ loader.batch_size=128
        >>>
        >>> # train model for 10 epochs
        >>> lightly-train input_dir=data/ trainer.max_epochs=10
        >>>
        >>> # print a full summary of the model
        >>> lightly-train input_dir=data/ trainer.weights_summary=full

    """
    return _train_cli(cfg)


def entry():
    train_cli()
