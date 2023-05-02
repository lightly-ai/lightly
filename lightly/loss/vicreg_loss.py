import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from lightly.utils.dist import gather


class VICRegLoss(torch.nn.Module):
    """Implementation of the VICReg loss [0].

    This implementation is based on the code published by the authors [1].

    - [0] VICReg, 2022, https://arxiv.org/abs/2105.04906
    - [1] https://github.com/facebookresearch/vicreg/

    Attributes:
        lambda_param:
            Scaling coefficient for the invariance term of the loss.
        mu_param:
            Scaling coefficient for the variance term of the loss.
        nu_param:
            Scaling coefficient for the covariance term of the loss.
        gather_distributed:
            If True then the cross-correlation matrices from all gpus are gathered and
            summed before the loss calculation.
        eps:
            Epsilon for numerical stability.

    Examples:

        >>> # initialize loss function
        >>> loss_fn = VICRegLoss()
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
        lambda_param: float = 25.0,
        mu_param: float = 25.0,
        nu_param: float = 1.0,
        gather_distributed: bool = False,
        eps=0.0001,
    ):
        super(VICRegLoss, self).__init__()
        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.gather_distributed = gather_distributed
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Returns VICReg loss.

        Args:
            z_a:
                Tensor with shape (batch_size, ..., dim).
            z_b:
                Tensor with shape (batch_size, ..., dim).
        """
        assert (
            z_a.shape[0] > 1 and z_b.shape[0] > 1
        ), f"z_a and z_b must have batch size > 1 but found {z_a.shape[0]} and {z_b.shape[0]}"
        assert (
            z_a.shape == z_b.shape
        ), f"z_a and z_b must have same shape but found {z_a.shape} and {z_b.shape}."

        # invariance term of the loss
        inv_loss = invariance_loss(x=z_a, y=z_b)

        # gather all batches
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                z_a = torch.cat(gather(z_a), dim=0)
                z_b = torch.cat(gather(z_b), dim=0)

        var_loss = 0.5 * (
            variance_loss(x=z_a, eps=self.eps) + variance_loss(x=z_b, eps=self.eps)
        )
        cov_loss = covariance_loss(x=z_a) + covariance_loss(x=z_b)

        loss = (
            self.lambda_param * inv_loss
            + self.mu_param * var_loss
            + self.nu_param * cov_loss
        )
        return loss


def invariance_loss(x: Tensor, y: Tensor) -> Tensor:
    """Returns VICReg invariance loss.

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
        y:
            Tensor with shape (batch_size, ..., dim).
    """
    return F.mse_loss(x, y)


def variance_loss(x: Tensor, eps: float = 0.0001) -> Tensor:
    """Returns VICReg variance loss.

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
        eps:
            Epsilon for numerical stability.
    """
    x = x - x.mean(dim=0)
    std = torch.sqrt(x.var(dim=0) + eps)
    loss = torch.mean(F.relu(1.0 - std))
    return loss


def covariance_loss(x: Tensor) -> Tensor:
    """Returns VICReg covariance loss.

    Generalized version of the covariance loss with support for tensors with more than
    two dimensions. Adapted from VICRegL:
    https://github.com/facebookresearch/VICRegL/blob/803ae4c8cd1649a820f03afb4793763e95317620/main_vicregl.py#L299

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
    """
    x = x - x.mean(dim=0)
    batch_size = x.size(0)
    dim = x.size(-1)
    # nondiag_mask has shape (dim, dim) with 1s on all non-diagonal entries.
    nondiag_mask = ~torch.eye(dim, device=x.device, dtype=torch.bool)
    # cov has shape (..., dim, dim)
    cov = torch.einsum("b...c,b...d->...cd", x, x) / (batch_size - 1)
    loss = cov[..., nondiag_mask].pow(2).sum(-1) / dim
    return loss.mean()
