

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
        new_key = '.'.join(key.split('.')[1:])
        new_state_dict[new_key] = item

    return new_state_dict


class _StateDictLoaderMixin:
    """Mixin which enables a common checkpoint loading interface.

    Filters the "model." prefix if necessary and expands batchnorm weights
    if the model uses SplitBatchNorm but the state dict contains default
    batchnorm weights.

    """

    def _custom_load_from_state_dict(self,
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
            _expand_batchnorm_weights(self.state_dict(), state_dict, self.num_splits)

        # step 3: load from checkpoint
        self.load_state_dict(state_dict, strict=strict)
