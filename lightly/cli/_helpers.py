""" Command-Line Interface Helpers """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
import torch
from hydra import utils

from lightly.models import ZOO as model_zoo


def fix_input_path(path):
    """Fix broken relative paths.

    """
    if not os.path.isabs(path):
        path = utils.to_absolute_path(path)
    return path


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


def _expand_batchnorm_weights(model_dict, state_dict, num_splits):
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


def _filter_state_dict(state_dict):
    """Prevents unexpected key error when loading PyTorch-Lightning checkpoints.

    Removes the "model." prefix from all keys in the state dictionary.

    """
    # 
    new_state_dict = {}
    for key, item in state_dict.items():
        key_parts = key.split('.')[1:]
        key_parts = [k if k != 'features' else 'backbone' for k in key_parts]
        new_key = '.'.join(key_parts)
        new_state_dict[new_key] = item

    return new_state_dict


def load_from_state_dict(model,
                         state_dict,
                         strict: bool = True,
                         apply_filter: bool = True):
    """Loads the model weights from the state dictionary.

    """
    # step 1: filter state dict
    if apply_filter:
        state_dict = _filter_state_dict(state_dict)
 
    # step 2: expand batchnorm weights
    state_dict = \
        _expand_batchnorm_weights(model.state_dict(), state_dict, 0)

    # step 3: load from checkpoint
    model.load_state_dict(state_dict, strict=strict)
