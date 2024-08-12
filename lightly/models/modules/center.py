from typing import Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module


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
        super().__init__()
        if mode not in CENTER_MODE_TO_FUNCTION:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid modes are "
                f"{sorted(CENTER_MODE_TO_FUNCTION.keys())}."
            )
        self._center_fn = CENTER_MODE_TO_FUNCTION[mode]

        self.size = size
        self.dim = tuple(i for i, s in enumerate(size) if s == 1)
        self.register_buffer("center", torch.zeros(self.size))
        self.momentum = momentum

    @property
    def value(self) -> Tensor:
        """The current value of the center. Use this property to do any operations based
        on the center."""
        return self.center

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        """Update the center with a new batch of features.

        Args:
            x:
                Feature tensor used to update the center. Must have the same number of
                dimensions as self.size.
        """
        batch_center = self._center_fn(x=x, dim=self.dim)
        self.center = center_momentum(
            center=self.center, batch_center=batch_center, momentum=self.momentum
        )

    @torch.no_grad()
    def _center_mean(self, x: Tensor) -> Tensor:
        """Returns the center of the input tensor by calculating the mean."""
        return center_mean(x=x, dim=self.dim)


@torch.no_grad()
def center_mean(x: Tensor, dim: Tuple[int, ...]) -> Tensor:
    """Returns the center of the input tensor by calculating the mean."""
    batch_center = torch.mean(x, dim=dim, keepdim=True)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(batch_center)
        batch_center = batch_center / dist.get_world_size()
    return batch_center


@torch.no_grad()
def center_momentum(center: Tensor, batch_center: Tensor, momentum: float) -> Tensor:
    """Returns the new center with momentum update."""
    return center * momentum + batch_center * (1 - momentum)


CENTER_MODE_TO_FUNCTION = {
    "mean": center_mean,
}
