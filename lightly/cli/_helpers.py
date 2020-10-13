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


def filter_state_dict(state_dict):
    """Prevent unexpected key error when loading PyTorch-Lightning checkpoints
       by removing the unnecessary prefix model. from each key.

    """
    new_state_dict = {}
    for key, item in state_dict.items():
        new_key = '.'.join(key.split('.')[1:])
        new_state_dict[new_key] = item
    return new_state_dict


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
