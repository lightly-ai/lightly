""" SplitBatchNorm Implementation """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


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

    def __init__(self, num_features: int, num_splits: int, **kw: Any) -> None:
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        # Register buffers
        self.register_buffer(
            "running_mean", torch.zeros(num_features * self.num_splits)
        )
        self.register_buffer("running_var", torch.ones(num_features * self.num_splits))

    def train(self, mode: bool = True) -> SplitBatchNorm:
        # lazily collate stats when we are going to use them
        if (self.training is True) and (mode is False):
            assert self.running_mean is not None
            self.running_mean = torch.mean(
                self.running_mean.view(self.num_splits, self.num_features), dim=0
            ).repeat(self.num_splits)
            assert self.running_var is not None
            self.running_var = torch.mean(
                self.running_var.view(self.num_splits, self.num_features), dim=0
            ).repeat(self.num_splits)

        return super().train(mode)

    def forward(self, input: Tensor) -> Tensor:
        """Computes the SplitBatchNorm on the input."""
        # get input shape
        N, C, H, W = input.shape

        # during training, use different stats for each split and otherwise
        # use the stats from the first split
        momentum = 0.0 if self.momentum is None else self.momentum
        if self.training or not self.track_running_stats:
            result = nn.functional.batch_norm(
                input=input.view(-1, C * self.num_splits, H, W),
                running_mean=self.running_mean,
                running_var=self.running_var,
                weight=self.weight.repeat(self.num_splits),
                bias=self.bias.repeat(self.num_splits),
                training=True,
                momentum=momentum,
                eps=self.eps,
            ).view(N, C, H, W)
        else:
            # We have to ignore the type errors here, because we know that running_mean
            # and running_var are not None, but the type checker does not.
            result = nn.functional.batch_norm(
                input=input,
                running_mean=self.running_mean[: self.num_features],  # type: ignore[index]
                running_var=self.running_var[: self.num_features],  # type: ignore[index]
                weight=self.weight,
                bias=self.bias,
                training=False,
                momentum=momentum,
                eps=self.eps,
            )

        return result


def get_norm_layer(num_features: int, num_splits: int, **kw: Any) -> nn.Module:
    """Utility to switch between BatchNorm2d and SplitBatchNorm."""
    if num_splits > 0:
        return SplitBatchNorm(num_features, num_splits)
    else:
        return nn.BatchNorm2d(num_features)
