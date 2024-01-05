import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module

from lightly.utils.dist import gather


class SSLEYLoss(Module):
    """Implementation of the SSL-EY loss [0].

    - [0]: Efficient Algorithms for the CCA Family: Unconstrained Objectives with Unbiased Gradients, 2023, https://arxiv.org/abs/2310.01012

    Attributes:
        gather_distributed:
            If True then the cross-correlation matrices from all gpus are gathered and
            summed before the loss calculation.
        eps:
            Epsilon for numerical stability.

    Examples:

        >>> # initialize loss function
        >>> loss_fn = SSLEYLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
    """

    def __init__(
        self,
        gather_distributed: bool = False,
        eps: float = 0.0001,
    ):
        super().__init__()
        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

        self.gather_distributed = gather_distributed
        self.eps = eps

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        """Returns SSL-EY loss.

        Args:
            z_a:
                Tensor with shape (batch_size, ..., dim).
            z_b:
                Tensor with shape (batch_size, ..., dim).
        """
        if z_a.shape[0] <= 1:
            raise ValueError(f"z_a must have batch size > 1 but found {z_a.shape[0]}.")
        if z_b.shape[0] <= 1:
            raise ValueError(f"z_b must have batch size > 1 but found {z_b.shape[0]}.")
        if z_a.shape != z_b.shape:
            raise ValueError(
                f"z_a and z_b must have same shape but found {z_a.shape} and "
                f"{z_b.shape}."
            )
        # gather all batches
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                z_a = torch.cat(gather(z_a), dim=0)
                z_b = torch.cat(gather(z_b), dim=0)

        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        batch_size = z_a.size(0)
        C = 2 * (z_a.T @ z_b) / (batch_size - 1)
        V = (z_a.T @ z_a) / (batch_size - 1) + (z_b.T @ z_b) / (batch_size - 1)

        loss = -2 * torch.trace(C) + torch.trace(V @ V)

        return loss
