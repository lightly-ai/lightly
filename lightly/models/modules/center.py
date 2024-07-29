from typing import Callable, Tuple

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
        _register_buffer:
            Deprecated, do not use. This argument is only kept for backwards
            compatibility with DINOLoss.
    """

    def __init__(
        self,
        size: Tuple[int, ...],
        mode: str = "mean",
        momentum: float = 0.9,
        _register_buffer: bool = True,
    ) -> None:
        super().__init__()

        modes = ["mean"]
        self._center_fn: Callable[[Tensor], Tensor]
        if mode not in modes:
            raise ValueError(f"Unknown mode '{mode}'. Valid modes are {modes}.")
        if mode == "mean":
            self._center_fn = self._center_mean
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented.")

        self.size = size
        self.dim = tuple(i for i, s in enumerate(size) if s == 1)

        center = torch.zeros(self.size)
        if _register_buffer:
            self.register_buffer("center", center)
        else:
            # Do not register buffer for backwards compatilibity with DINOLoss as the
            # loss already registers the buffer. If we register it here again there will
            # be an extra entry in the state dict.
            self.center = center

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
        batch_center = self._center_fn(x)
        # Use copy for backwards compatibility with DINOLoss.
        self.center.copy_(self._center_momentum(batch_center))

    @torch.no_grad()
    def _center_mean(self, x: Tensor) -> Tensor:
        """Returns the center of the input tensor by calculating the mean."""
        batch_center = torch.mean(x, dim=self.dim, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()
        return batch_center

    @torch.no_grad()
    def _center_momentum(self, batch_center: Tensor) -> Tensor:
        """Returns the new center with momentum update."""
        return self.center * self.momentum + batch_center * (1 - self.momentum)
