# -*- coding: utf-8 -*-
"""**Lightly Train:** Train a self-supervised model from the command-line.

This module contains the entrypoint for the **lightly-train**
command-line interface.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import hydra
import torch
import warnings

from lightly.data import ImageCollateFunction
from lightly.data import LightlyDataset
from lightly.embedding import SelfSupervisedEmbedding
from lightly.loss import NTXentLoss
from lightly.models import ResNetSimCLR

from lightly.cli._helpers import is_url
from lightly.cli._helpers import get_ptmodel_from_config
from lightly.cli._helpers import fix_input_path
from lightly.cli._helpers import load_state_dict_from_url


def _train_cli(cfg, is_cli_call=True):

    data = cfg['data']
    download = cfg['download']

    root = cfg['root']
    if root and is_cli_call:
        root = fix_input_path(root)

    input_dir = cfg['input_dir']
    if input_dir and is_cli_call:
        input_dir = fix_input_path(input_dir)

    if 'seed' in cfg.keys():
        seed = cfg['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = 'cuda'
    elif cfg['trainer'] and cfg['trainer']['gpus']:
        device = 'cpu'
        cfg['trainer']['gpus'] = 0

    if cfg['loader']['batch_size'] < 64:
        msg = 'Training a self-supervised model with a small batch size: {}! '
        msg = msg.format(cfg['loader']['batch_size'])
        msg += 'Small batch size may harm embedding quality. '
        msg += 'You can specify the batch size via the loader key-word: '
        msg += 'loader.batch_size=BSZ'
        warnings.warn(msg)

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
    if state_dict is not None:
        model = ResNetSimCLR.from_state_dict(state_dict, **cfg['model'])
    else:
        model = ResNetSimCLR(**cfg['model'])

    criterion = NTXentLoss(**cfg['criterion'])
    optimizer = torch.optim.SGD(model.parameters(), **cfg['optimizer'])

    dataset = LightlyDataset(root,
                           name=data, train=True, download=download,
                           from_folder=input_dir)

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
    encoder = encoder.train_embedding(**cfg['trainer'])

    print('Best model is stored at: %s' % (encoder.checkpoint))
    return encoder.checkpoint


@hydra.main(config_path="config", config_name="config")
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

    """
    return _train_cli(cfg)


def entry():
    train_cli()
