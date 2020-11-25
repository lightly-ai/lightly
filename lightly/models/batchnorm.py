""" SplitBatchNorm Implementation """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn


class SplitBatchNorm(nn.BatchNorm2d):
    """Simulates multi-gpu behaviour of BatchNorm in one gpu by splitting.

    Implementation was adapted from:
    https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py

    Attributes:
        num_features:
            Number of input features.
        num_splits:
            Number of splits.

    """
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer(
            'running_mean', torch.zeros(num_features*self.num_splits)
        )
        self.register_buffer(
            'running_var', torch.ones(num_features*self.num_splits)
        )

    def train(self, mode=True):
        # lazily collate stats when we are going to use them
        if (self.training is True) and (mode is False):
            self.running_mean = \
                torch.mean(
                    self.running_mean.view(self.num_splits, self.num_features),
                    dim=0
                ).repeat(self.num_splits)
            self.running_var = \
                torch.mean(
                    self.running_var.view(self.num_splits, self.num_features),
                    dim=0
                ).repeat(self.num_splits)

        return super().train(mode)

    def forward(self, input):
        """Computes the SplitBatchNorm on the input.

        """
        # get input shape
        N, C, H, W = input.shape

        # during training, use different stats for each split and otherwise
        # use the stats from the first split
        if self.training or not self.track_running_stats:
            result = nn.functional.batch_norm(
                input.view(-1, C*self.num_splits, H, W),
                self.running_mean, self.running_var, 
                self.weight.repeat(self.num_splits),
                self.bias.repeat(self.num_splits),
                True,
                self.momentum,
                self.eps
            ).view(N, C, H, W)
        else:
            result = nn.functional.batch_norm(
                input,
                self.running_mean[:self.num_features],
                self.running_var[:self.num_features], 
                self.weight,
                self.bias,
                False,
                self.momentum, 
                self.eps
            )
        
        return result


def get_norm_layer(num_features: int, num_splits: int, **kw):
    """Utility to switch between BatchNorm2d and SplitBatchNorm.

    """
    if num_splits > 0:
        return SplitBatchNorm(num_features, num_splits)
    else:
        return nn.BatchNorm2d(num_features)
