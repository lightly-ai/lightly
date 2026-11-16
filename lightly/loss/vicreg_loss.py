from typing import Any, Optional, Tuple

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
            If True, the cross-correlation matrices from all GPUs are gathered and
            summed before the loss calculation.
        eps:
            Epsilon for numerical stability.
        covariance_block_size:
            Compute and recompute covariance rows in blocks during forward and
            backward to bound temporary memory for wide projection heads.

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
        eps: float = 0.0001,
        covariance_block_size: Optional[int] = None,
    ):
        """Initializes the VICRegLoss module with the specified parameters.

        Raises:
            ValueError: If gather_distributed is True but torch.distributed is not available.
        """
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
        self.covariance_block_size = covariance_block_size

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Returns VICReg loss.

        Args:
            z_a:
                Tensor with shape (batch_size, ..., dim).
            z_b:
                Tensor with shape (batch_size, ..., dim).

        Returns:
            The computed VICReg loss.

        Raises:
            AssertionError: If z_a or z_b have a batch size <= 1.
            AssertionError: If z_a and z_b do not have the same shape.
        """
        assert z_a.shape[0] > 1 and z_b.shape[0] > 1, (
            f"z_a and z_b must have batch size > 1 but found {z_a.shape[0]} and {z_b.shape[0]}"
        )
        assert z_a.shape == z_b.shape, (
            f"z_a and z_b must have same shape but found {z_a.shape} and {z_b.shape}."
        )

        # Invariance term of the loss
        inv_loss = invariance_loss(x=z_a, y=z_b)

        # Gather all batches
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                z_a = torch.cat(gather(z_a), dim=0)
                z_b = torch.cat(gather(z_b), dim=0)

        # Variance and covariance terms of the loss
        var_loss = 0.5 * (
            variance_loss(x=z_a, eps=self.eps) + variance_loss(x=z_b, eps=self.eps)
        )
        cov_loss = covariance_loss(z_a, self.covariance_block_size) + covariance_loss(
            z_b, self.covariance_block_size
        )

        # Total VICReg loss
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

    Returns:
        The computed VICReg invariance loss.
    """
    return F.mse_loss(x, y)


def variance_loss(x: Tensor, eps: float = 0.0001) -> Tensor:
    """Returns VICReg variance loss.

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
        eps:
            Epsilon for numerical stability.

    Returns:
        The computed VICReg variance loss.
    """
    std = torch.sqrt(x.var(dim=0) + eps)
    loss = torch.mean(F.relu(1.0 - std))
    return loss


def _covariance_rows(x: Tensor, start: int, stop: int) -> Tensor:
    cov = torch.einsum("b...c,b...d->...cd", x[..., start:stop], x) / (x.size(0) - 1)
    cov.diagonal(offset=start, dim1=-2, dim2=-1).zero_()
    return cov


class _BlockedCovariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, block_size: int) -> Tensor:
        ctx.save_for_backward(x)
        ctx.block_size = block_size
        total = x.new_zeros(x.shape[1:-1])
        for start in range(0, x.size(-1), block_size):
            rows = _covariance_rows(x, start, start + block_size)
            total = total + rows.square().sum(dim=(-2, -1))
        ctx.cov_dtype = rows.dtype
        return total

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None]:
        (x,) = ctx.saved_tensors
        x = x.to(ctx.cov_dtype)
        # For centered X and covariance C, the gradient of the squared
        # off-diagonal norm is 4 X offdiag(C) / (N - 1).
        scale = grad_output[..., None, None] * (2 / (x.size(0) - 1))
        parts = []
        for start in range(0, x.size(-1), ctx.block_size):
            rows = _covariance_rows(x, start, start + ctx.block_size)
            # Apply the upstream loss scale before the low-precision product.
            scaled_rows = (rows * scale).to(x.dtype)
            parts.append(2 * torch.einsum("b...d,...cd->b...c", x, scaled_rows))
        return torch.cat(parts, dim=-1), None


def covariance_loss(x: Tensor, block_size: Optional[int] = None) -> Tensor:
    """Returns VICReg covariance loss.

    Generalized version of the covariance loss with support for tensors with more than
    two dimensions. Adapted from VICRegL:
    https://github.com/facebookresearch/VICRegL/blob/803ae4c8cd1649a820f03afb4793763e95317620/main_vicregl.py#L299

    Args:
        x: Tensor with shape (batch_size, ..., dim).
        block_size: Optional maximum number of covariance rows computed at once.

    Returns:
          The computed VICReg covariance loss.
    """
    x = x - x.mean(dim=0)
    batch_size = x.size(0)
    dim = x.size(-1)
    if block_size is not None:
        if block_size < 1:
            raise ValueError("covariance block size must be positive")
        if block_size < dim:
            return (_BlockedCovariance.apply(x, block_size) / dim).mean()

    # nondiag_mask has shape (dim, dim) with 1s on all non-diagonal entries.
    nondiag_mask = ~torch.eye(dim, device=x.device, dtype=torch.bool)

    # cov has shape (..., dim, dim)
    cov = torch.einsum("b...c,b...d->...cd", x, x) / (batch_size - 1)

    loss = cov[..., nondiag_mask].pow(2).sum(-1) / dim
    return loss.mean()
