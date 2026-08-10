from typing import Tuple, Union

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

    @property
    def value(self) -> Tensor:
        """The current value of the center.

        Use this property to do any operations based on the center.
        """
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


@torch.no_grad()
def sinkhorn_knopp(
    x: Tensor,
    temperature: Union[float, Tensor] = 0.04,
    num_iterations: int = 3,
    gather_distributed: bool = True,
) -> Tensor:
    """Returns Sinkhorn-Knopp normalized probabilities as introduced in SwAV [0].

    Instead of subtracting a running center, the sharpened logits are normalized such
    that every prototype receives the same total weight across the batch, which avoids
    collapse without tracking any state. DINOv2 [1] offers this as an alternative to the
    mean centering of DINO [2], but keeps mean centering in its released configs, where
    it reports no difference on ImageNet-1k. DINOv3 [3] uses it for both the DINO and
    the iBOT objective.

    Implementation is based on [4] and shared with lightly.loss.swav_loss.sinkhorn.
    The number of samples is reduced across processes instead of being derived from the
    world size, because processes can hold a different number of samples. This is the
    case for the iBOT loss, where the number of masked tokens differs between processes.

    - [0]: SwAV, 2020, https://arxiv.org/abs/2006.09882
    - [1]: DINOv2, 2023, https://arxiv.org/abs/2304.07193
    - [2]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [3]: DINOv3, 2025, https://arxiv.org/abs/2508.10104
    - [4]: https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/dino_clstoken_loss.py

    Args:
        x:
            Tensor with shape (batch_size, num_prototypes) containing the logits.
        temperature:
            Temperature used to sharpen the logits.
        num_iterations:
            Number of Sinkhorn-Knopp iterations.
        gather_distributed:
            If True, the normalization is computed over the batches of all processes.

    Returns:
        Tensor with shape (batch_size, num_prototypes) containing the probabilities in
        float32. Every row sums to one if num_iterations is at least one. The
        probabilities are detached from the computation graph, following the reference
        implementation.
    """
    gather = gather_distributed and dist.is_available() and dist.is_initialized()

    # (batch_size, num_prototypes) -> (num_prototypes, batch_size) following the
    # notation of the reference implementation. The exponential is computed in float32
    # because it overflows in half precision for the low temperatures used by DINO.
    Q = torch.exp(x.float() / temperature).t()
    num_prototypes = Q.shape[0]

    # The number of samples is reduced together with the sum over Q to save a
    # synchronization point. The actual number of samples is reduced instead of
    # multiplying the local batch size with the world size because processes can have
    # different batch sizes. This happens for example in the iBOT loss where the number
    # of masked tokens differs between processes.
    local_num_samples = torch.tensor(Q.shape[1], device=Q.device, dtype=Q.dtype)
    sums = torch.stack([Q.sum(), local_num_samples])
    if gather:
        dist.all_reduce(sums)
    sum_Q, num_samples = sums[0], sums[1]

    # Make the matrix sum to 1.
    Q /= sum_Q

    for _ in range(num_iterations):
        # Normalize rows: the total weight per prototype must be 1 / num_prototypes.
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if gather:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= num_prototypes

        # Normalize columns: the total weight per sample must be 1 / num_samples.
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= num_samples

    # Scale the columns to sum to one, such that Q is an assignment.
    Q *= num_samples
    return Q.t()


CENTER_MODE_SINKHORN_KNOPP = "sinkhorn_knopp"

CENTER_MODE_TO_FUNCTION = {
    "mean": center_mean,
}

# Modes accepted by the losses. The Center module only supports the modes in
# CENTER_MODE_TO_FUNCTION, as Sinkhorn-Knopp does not track a center.
VALID_CENTER_MODES = [*CENTER_MODE_TO_FUNCTION, CENTER_MODE_SINKHORN_KNOPP]
