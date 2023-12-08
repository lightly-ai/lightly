import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from lightly.utils.dist import gather


class SSLEYLoss(torch.nn.Module):
    """Implementation of the SSL-EY loss [0].

    - [0] Efficient Algorithms for the CCA Family: Unconstrained Objectives with Unbiased Gradients, 2022, https://arxiv.org/abs/2310.01012

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
        eps=0.0001,
    ):
        super(SSLEYLoss, self).__init__()
        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

        self.gather_distributed = gather_distributed
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Returns SSL-EY loss.

        Args:
            z_a:
                Tensor with shape (batch_size, ..., dim).
            z_b:
                Tensor with shape (batch_size, ..., dim).
        """

        # gather all batches
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                z_a = torch.cat(gather(z_a), dim=0)
                z_b = torch.cat(gather(z_b), dim=0)

        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)

        C = 2*(z_a.T @ z_b) / (self.args.batch_size - 1)
        V = (z_a.T @ z_a) / (self.args.batch_size - 1) + (z_b.T @ z_b) / (self.args.batch_size - 1)

        loss = torch.trace(C)-torch.trace(V@V)

        return loss
