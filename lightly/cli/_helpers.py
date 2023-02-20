""" Command-Line Interface Helpers """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
import os

import torch
import hydra
from hydra import utils
from torch import nn as nn

from lightly.utils.version_compare import version_compare

from lightly.cli._cli_simclr import _SimCLR
from lightly.embedding import SelfSupervisedEmbedding

from lightly.models import ZOO as model_zoo, ResNetGenerator
from lightly.models.batchnorm import get_norm_layer






def cpu_count():
    """Returns the number of CPUs which are present in the system.

    This number is not equivalent to the number of available CPUs to the process.

    """
    return os.cpu_count()


def fix_input_path(path):
    """Fix broken relative paths.

    """
    if not os.path.isabs(path):
        path = utils.to_absolute_path(path)
    return path


def fix_hydra_arguments(config_path: str = 'config', config_name: str = 'config'):
    """Helper to make hydra arugments adaptive to installed hydra version
    
    Hydra introduced the `version_base` argument in version 1.2.0
    We use this helper to provide backwards compatibility to older hydra verisons.    
    """

    hydra_args = {'config_path': config_path, 'config_name': config_name}

    try:
        if version_compare(hydra.__version__, '1.1.2') > 0:
            hydra_args['version_base'] = '1.1'
    except ValueError:
        pass
    
    return hydra_args


def is_url(checkpoint):
    """Check whether the checkpoint is a url or not.

    """
    is_url = ('https://storage.googleapis.com' in checkpoint)
    return is_url


def get_ptmodel_from_config(model):
    """Get a pre-trained model from the lightly model zoo.

    """
    key = model['name']
    key += '/simclr'
    key += '/d' + str(model['num_ftrs'])
    key += '/w' + str(float(model['width']))

    if key in model_zoo.keys():
        return model_zoo[key], key
    else:
        return '', key


def load_state_dict_from_url(url, map_location=None):
    """Try to load the checkopint from the given url.

    """
    try:
        state_dict = torch.hub.load_state_dict_from_url(
            url, map_location=map_location
        )
        return state_dict
    except Exception:
        print('Not able to load state dict from %s' % (url))
        print('Retrying with http:// prefix')
    try:
        url = url.replace('https', 'http')
        state_dict = torch.hub.load_state_dict_from_url(
            url, map_location=map_location
        )
        return state_dict
    except Exception:
        print('Not able to load state dict from %s' % (url))

    # in this case downloading the pre-trained model was not possible
    # notify the user and return
    return {'state_dict': None}


def _maybe_expand_batchnorm_weights(model_dict, state_dict, num_splits):
    """Expands the weights of the BatchNorm2d to the size of SplitBatchNorm.

    """
    running_mean = 'running_mean'
    running_var = 'running_var'

    for key, item in model_dict.items():
        # not batchnorm -> continue
        if not running_mean in key and not running_var in key:
            continue

        state = state_dict.get(key, None)
        # not in dict -> continue
        if state is None:
            continue
        # same shape -> continue
        if item.shape == state.shape:
            continue

        # found running mean or running var with different shapes
        state_dict[key] = state.repeat(num_splits)

    return state_dict


def _filter_state_dict(state_dict, remove_model_prefix_offset: int = 1):
    """Makes the state_dict compatible with the model.
    
    Prevents unexpected key error when loading PyTorch-Lightning checkpoints.
    Allows backwards compatability to checkpoints before v1.0.6.

    """

    prev_backbone = 'features'
    curr_backbone = 'backbone'

    new_state_dict = {}
    for key, item in state_dict.items():
        # remove the "model." prefix from the state dict key
        key_parts = key.split('.')[remove_model_prefix_offset:]
        # with v1.0.6 the backbone of the models will be renamed from
        # "features" to "backbone", ensure compatability with old ckpts
        key_parts = \
            [k if k != prev_backbone else curr_backbone for k in key_parts]

        new_key = '.'.join(key_parts)
        new_state_dict[new_key] = item

    return new_state_dict


def _fix_projection_head_keys(state_dict):
    """Makes the state_dict compatible with the refactored projection heads.

    TODO: Remove once the models are refactored and the old checkpoints were
    replaced! Relevant issue: https://github.com/lightly-ai/lightly/issues/379

    Prevents unexpected key error when loading old checkpoints.
    
    """

    projection_head_identifier = 'projection_head'
    prediction_head_identifier = 'prediction_head'
    projection_head_insert = 'layers'

    new_state_dict = {}
    for key, item in state_dict.items():
        if (projection_head_identifier in key or \
            prediction_head_identifier in key) and \
                projection_head_insert not in key:
            # insert layers if it's not part of the key yet
            key_parts = key.split('.')
            key_parts.insert(1, projection_head_insert)
            new_key = '.'.join(key_parts)
        else:
            new_key = key

        new_state_dict[new_key] = item

    return new_state_dict


def load_from_state_dict(model,
                         state_dict,
                         strict: bool = True,
                         apply_filter: bool = True,
                         num_splits: int = 0):
    """Loads the model weights from the state dictionary.

    """

    # step 1: filter state dict
    if apply_filter:
        state_dict = _filter_state_dict(state_dict)

    state_dict = _fix_projection_head_keys(state_dict)

    # step 2: expand batchnorm weights
    state_dict = \
        _maybe_expand_batchnorm_weights(model.state_dict(), state_dict, num_splits)

    # step 3: load from checkpoint
    model.load_state_dict(state_dict, strict=strict)


def get_model_from_config(cfg, is_cli_call: bool = False) -> SelfSupervisedEmbedding:
    checkpoint = cfg['checkpoint']
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if not checkpoint:
        checkpoint, key = get_ptmodel_from_config(cfg['model'])
        if not checkpoint:
            msg = 'Cannot download checkpoint for key {} '.format(key)
            msg += 'because it does not exist!'
            raise RuntimeError(msg)
        state_dict = load_state_dict_from_url(checkpoint, map_location=device)[
            'state_dict'
        ]
    else:
        checkpoint = fix_input_path(checkpoint) if is_cli_call else checkpoint
        state_dict = torch.load(checkpoint, map_location=device)['state_dict']

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
        features, num_ftrs=cfg['model']['num_ftrs'], out_dim=cfg['model']['out_dim']
    ).to(device)

    if state_dict is not None:
        load_from_state_dict(model, state_dict)

    encoder = SelfSupervisedEmbedding(model, None, None, None)
    return encoder
