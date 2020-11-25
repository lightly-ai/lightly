# -*- coding: utf-8 -*-
"""**Lightly Embed:** Embed images with one command.

This module contains the entrypoint for the **lightly-embed**
command-line interface.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os

import hydra
import torch
import torchvision

from lightly.data import LightlyDataset
from lightly.embedding import SelfSupervisedEmbedding
from lightly.models import ResNetSimCLR
from lightly.utils import save_embeddings

from lightly.cli._helpers import get_ptmodel_from_config
from lightly.cli._helpers import fix_input_path
from lightly.cli._helpers import load_state_dict_from_url


def _embed_cli(cfg, is_cli_call=True):

    data = cfg['data']
    train = cfg.get('train', True)
    checkpoint = cfg['checkpoint']
    download = cfg['download']

    root = cfg['root']
    if root and is_cli_call:
        root = fix_input_path(root)

    input_dir = cfg['input_dir']
    if input_dir and is_cli_call:
        input_dir = fix_input_path(input_dir)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((cfg['collate']['input_size'],
                                       cfg['collate']['input_size'])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    dataset = LightlyDataset(root,
                           name=data, train=train, download=download,
                           from_folder=input_dir, transform=transform)

    cfg['loader']['drop_last'] = False
    cfg['loader']['shuffle'] = False
    cfg['loader']['batch_size'] = min(
        cfg['loader']['batch_size'],
        len(dataset)
    )
    dataloader = torch.utils.data.DataLoader(dataset, **cfg['loader'])

    # load the PyTorch state dictionary and map it to the current device    
    state_dict = None
    if not checkpoint:
        checkpoint, key = get_ptmodel_from_config(cfg['model'])
        if not checkpoint:
            msg = 'Cannot download checkpoint for key {} '.format(key)
            msg += 'because it does not exist!'
            raise RuntimeError(msg)
        state_dict = load_state_dict_from_url(
            checkpoint, map_location=device
        )['state_dict']
    else:
        checkpoint = fix_input_path(checkpoint) if is_cli_call else checkpoint
        state_dict = torch.load(
            checkpoint, map_location=device
        )['state_dict']

    model = ResNetSimCLR(**cfg['model']).to(device)
    if state_dict is not None:
        model.load_from_state_dict(state_dict)

    encoder = SelfSupervisedEmbedding(model, None, None, None)
    embeddings, labels, filenames = encoder.embed(dataloader, device=device)

    if is_cli_call:
        path = os.path.join(os.getcwd(), 'embeddings.csv')
        save_embeddings(path, embeddings, labels, filenames)
        print('Embeddings are stored at %s' % (path))
        return path

    return embeddings, labels, filenames


@hydra.main(config_path='config', config_name='config')
def embed_cli(cfg):
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
