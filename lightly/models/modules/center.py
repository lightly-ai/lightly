from __future__ import annotations

import math
import warnings
from typing import Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module

# Number of accumulations after which a missing momentum update is reported.
MAX_ACCUMULATED = 1000


class Center(Module):
    """Center module to compute and store the center of a feature tensor as used
    in DINO [0].

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294

    Attributes:
        size:
            Size of the tracked center tensor. Dimensions across which the center
            is computed must be set to 1. For example, if the feature tensor has shape
            (batch_size, sequence_length, feature_dim) and the center should be computed
            across the batch and sequence dimensions, the size should be
            (1, 1, feature_dim).
        mode:
            Mode to compute the center. Currently only 'mean' is supported.
        momentum:
            Momentum term for the center calculation.
    """

    def __init__(
        self,
        size: Tuple[int, ...],
        mode: str = "mean",
        momentum: float = 0.9,
    ) -> None:
        """Initializes the Center module with the specified parameters.

        Raises:
            ValueError: If an unknown mode is provided.
        """
        super().__init__()

        if mode not in CENTER_MODE_TO_FUNCTION:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid modes are "
                f"{sorted(CENTER_MODE_TO_FUNCTION.keys())}."
            )

        self.size = size
        self.dim = tuple(i for i, s in enumerate(size) if s == 1)
        self.center: Tensor  # For mypy
        self.register_buffer("center", torch.zeros(self.size))
        self.momentum = momentum

        # Plain attributes instead of buffers: DDP broadcasts buffers from rank
        # zero before every forward, which would overwrite these rank-local sums.
        self._batch_sum: Tensor | None = None
        self._batch_num_elements = 0
        self._num_accumulated = 0
        self._warned_missing_update = False

    @property
    def value(self) -> Tensor:
        """The current value of the center.

        Use this property to do any operations based on the center.
        """
        return self.center

    @torch.no_grad()
    def update(self, x: Tensor | None = None) -> None:
        """Update the center with a new batch of features.

        Args:
            x:
                Feature tensor used to update the center. Must have the same number of
                dimensions as self.size. If None, the center is updated from the
                features passed to previous accumulate calls only.
        """
        if x is not None:
            self.accumulate(x)
        self.apply_update()

    @torch.no_grad()
    def accumulate(self, x: Tensor) -> None:
        """Accumulate a new batch of features without updating the center.

        Use together with apply_update to apply a single momentum update per
        optimizer step when training with gradient accumulation.

        Args:
            x:
                Feature tensor used to update the center. Must have the same number of
                dimensions as self.size.
        """
        batch_sum = torch.sum(x, dim=self.dim, keepdim=True)
        if self._batch_sum is None:
            self._batch_sum = batch_sum
        else:
            self._batch_sum += batch_sum
        self._batch_num_elements += math.prod(x.shape[d] for d in self.dim)
        self._num_accumulated += 1

        if self._num_accumulated > MAX_ACCUMULATED and not self._warned_missing_update:
            self._warned_missing_update = True
            warnings.warn(
                f"{type(self).__name__} accumulated {self._num_accumulated} batch "
                "centers without a call to apply_update(). If you defer the center "
                "update for gradient accumulation, you must apply it once per "
                "optimizer step, otherwise the center stays frozen and the model "
                "may collapse.",
                UserWarning,
                stacklevel=2,
            )

    @torch.no_grad()
    def apply_update(self) -> None:
        """Update the center from the accumulated features and reset them.

        Does nothing if no features were accumulated.

        Runs a distributed collective, so all ranks must call this in lockstep.
        """
        if self._batch_sum is None:
            return
        batch_center = reduce_mean(self._batch_sum / self._batch_num_elements)
        self.center = center_momentum(
            center=self.center, batch_center=batch_center, momentum=self.momentum
        )
        self._batch_sum = None
        self._batch_num_elements = 0
        self._num_accumulated = 0


@torch.no_grad()
def center_mean(x: Tensor, dim: Tuple[int, ...]) -> Tensor:
    """Returns the center of the input tensor by calculating the mean.

    Args:
        x:
            Input tensor.
        dim:
            Dimensions along which the mean is calculated.

    Returns:
        The center of the input tensor.
    """
    return reduce_mean(torch.mean(x, dim=dim, keepdim=True))


@torch.no_grad()
def reduce_mean(x: Tensor) -> Tensor:
    """Returns the mean of the input tensor across all processes.

    The input tensor is reduced in place.

    Args:
        x:
            Input tensor. Must have the same shape on all processes.

    Returns:
        The mean of the input tensor across all processes.
    """
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x)
        x = x / dist.get_world_size()
    return x


@torch.no_grad()
def center_momentum(center: Tensor, batch_center: Tensor, momentum: float) -> Tensor:
    """Returns the new center with momentum update."""
    return center * momentum + batch_center * (1 - momentum)


CENTER_MODE_TO_FUNCTION = {
    "mean": center_mean,
}
