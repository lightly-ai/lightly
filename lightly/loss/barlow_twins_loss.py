from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


class BarlowTwinsLoss(torch.nn.Module):
    """Implementation of the Barlow Twins Loss from Barlow Twins[0] paper.
    This code specifically implements the Figure Algorithm 1 from [0].

    [0] Zbontar,J. et.al, 2021, Barlow Twins... https://arxiv.org/abs/2103.03230

        Examples:

        >>> # initialize loss function
        >>> loss_fn = BarlowTwinsLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimSiam model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)

    """

    def __init__(self, lambda_param: float = 5e-3, gather_distributed: bool = False):
        """Lambda param configuration with default value like in [0]

        Args:
            lambda_param:
                Parameter for importance of redundancy reduction term.
                Defaults to 5e-3 [0].
            gather_distributed:
                If True then the cross-correlation matrices from all gpus are
                gathered and summed before the loss calculation.
        """
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed

        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        # normalize repr. along the batch dimension
        z_a_norm, z_b_norm = _normalize(z_a, z_b)

        N = z_a.size(0)

        # cross-correlation matrix
        c = z_a_norm.T @ z_b_norm
        c.div_(N)

        # sum cross-correlation matrix between multiple gpus
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                c = c / world_size
                dist.all_reduce(c)

        invariance_loss = torch.diagonal(c).add_(-1).pow_(2).sum()
        redundancy_reduction_loss = _off_diagonal(c).pow_(2).sum()
        loss = invariance_loss + self.lambda_param * redundancy_reduction_loss

        return loss


def _normalize(
    z_a: torch.Tensor, z_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper function to normalize tensors along the batch dimension."""
    combined = torch.stack([z_a, z_b], dim=0)  # Shape: 2 x N x D
    normalized = F.batch_norm(
        combined.flatten(0, 1),
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        training=True,
    ).view_as(combined)
    return normalized[0], normalized[1]


def _off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
