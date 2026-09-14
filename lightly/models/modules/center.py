from __future__ import annotations

import warnings
from typing import Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module

# Number of accumulated batch centers after which we warn that the momentum update
# was never applied. Chosen far above any realistic gradient accumulation factor.
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

        center_fn = CENTER_MODE_TO_FUNCTION.get(mode)
        if center_fn is None:
            raise ValueError(
                f"Unknown mode '{mode}'. Valid modes are "
                f"{sorted(CENTER_MODE_TO_FUNCTION.keys())}."
            )
        self._center_fn = center_fn

        self.size = size
        self.dim = tuple(i for i, s in enumerate(size) if s == 1)
        self.center: Tensor  # For mypy
        self.register_buffer("center", torch.zeros(self.size))
        self.momentum = momentum

        # Batch centers accumulated since the last momentum update. The buffer is
        # non-persistent so that state dicts stay compatible with checkpoints that
        # were written before accumulation was introduced.
        self._batch_center_sum: Tensor  # For mypy
        self.register_buffer(
            "_batch_center_sum", torch.zeros(self.size), persistent=False
        )
        # Kept as a plain Python int on purpose. A tensor counter would force a
        # device synchronization in apply_update. Note that this makes accumulate
        # and apply_update unsuitable for a torch.compile'd region; the
        # dist.all_reduce in center_mean causes a graph break there anyway.
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
        # NOTE(Lionel, 09/26): The distributed all-reduce happens here, inside
        # accumulate, which makes the accumulated sum identical on all ranks. DDP
        # broadcasts buffers (including non-persistent ones) from rank zero before
        # every forward pass, so that broadcast is a no-op for the accumulator.
        # Moving the all-reduce to apply_update would break this.
        batch_center = self._center_fn(x=x, dim=self.dim)
        self._batch_center_sum += batch_center
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
        """
        if self._num_accumulated == 0:
            return
        batch_center = self._batch_center_sum / self._num_accumulated
        self.center = center_momentum(
            center=self.center, batch_center=batch_center, momentum=self.momentum
        )
        self._batch_center_sum.zero_()
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
