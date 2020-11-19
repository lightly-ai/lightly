from difflib import ndiff


def filter_state_dict(state_dict):
    """Prevent unexpected key error when loading PyTorch-Lightning checkpoints
       by removing the unnecessary prefix model. from each key.

    """
    new_state_dict = {}
    for key, item in state_dict.items():
        new_key = '.'.join(key.split('.')[1:])
        new_state_dict[new_key] = item
    return new_state_dict