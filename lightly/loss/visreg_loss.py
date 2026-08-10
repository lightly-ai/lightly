"""VISReg loss functions."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch import distributed as torch_dist
from torch.distributions import Normal

from lightly.loss import lejepa_loss
from lightly.utils import dist as lightly_dist


def visreg_center_loss(z: Tensor) -> Tensor:
    """VISReg center loss penalizing the batch mean of the embeddings.

    Pushes the per-dimension batch mean of the embeddings toward zero so
    the embedding distribution stays centered at the origin (Eq. 6 in the
    paper).

    Reference:
        VISReg, Wu et al., 2026, https://arxiv.org/abs/2606.02572

    Args:
        z: Projected embeddings of shape ``(..., N, D)`` where ``N`` is the
            batch size and ``D`` is the projection dimensionality.

    Returns:
        Scalar center loss.
    """
    return z.mean(dim=-2).square().mean()


def visreg_scale_loss(z: Tensor) -> Tensor:
    """VISReg scale loss regulating the per-dimension standard deviation.

    Penalizes the squared deviation of every embedding dimension's standard
    deviation from one (Eq. 1 in the paper). The gradient of this term
    approaches a constant as the embeddings collapse, providing a
    corrective signal exactly when characteristic-function sketching loses
    it (Figure 2 in the paper).

    Reference:
        VISReg, Wu et al., 2026, https://arxiv.org/abs/2606.02572

    Args:
        z: Projected embeddings of shape ``(..., N, D)`` where ``N`` is the
            batch size and ``D`` is the projection dimensionality.

    Returns:
        Scalar scale loss.
    """
    std = z.std(dim=-2, unbiased=False)
    loss: Tensor = (1.0 - std).square().mean()
    return loss


def visreg_shape_loss(z: Tensor, num_slices: int, eps: float = 1e-4) -> Tensor:
    """VISReg shape loss sketching the embeddings toward an isotropic Gaussian.

    Implements the sliced-Wasserstein sketching term (Eq. 5 in the paper).
    The embeddings are centered and normalized by their standard deviation
    with a stop-gradient on the standard deviation, so shape optimization
    does not interfere with scale regulation (Eq. 2 in the paper). The
    normalized embeddings are projected onto ``num_slices`` random unit
    directions and the sorted projections are compared against the
    quantiles of a standard Gaussian, which is the closed-form 1D
    2-Wasserstein distance.

    Every call draws fresh random directions from the default generator. In
    distributed training each device therefore uses its own slices, which
    multiplies the effective number of slices by the world size (Section
    3.2 in the paper) provided the random number generators of the devices
    are seeded differently.

    Reference:
        VISReg, Wu et al., 2026, https://arxiv.org/abs/2606.02572

    Args:
        z: Projected embeddings of shape ``(..., N, D)`` where ``N`` is the
            batch size and ``D`` is the projection dimensionality.
        num_slices: Number of random 1D projection directions (slices).
        eps: Numerical stability term added to the standard deviation before
            normalization.

    Returns:
        Scalar shape loss.

    Raises:
        ValueError: If num_slices or eps is not positive.
    """
    if num_slices < 1:
        raise ValueError("num_slices must be a positive integer.")
    if eps <= 0:
        raise ValueError("eps must be greater than zero.")

    num_samples = z.size(-2)
    num_features = z.size(-1)

    mean = z.mean(dim=-2, keepdim=True)
    std = z.std(dim=-2, unbiased=False, keepdim=True)
    # The stop-gradient on std decouples shape from scale optimization.
    z_norm = (z - mean) / (std.detach() + eps)

    slices = torch.randn(num_features, num_slices, device=z.device, dtype=z.dtype)
    slices = slices / slices.norm(p=2, dim=0)
    projections = z_norm @ slices

    # Fixed standard Gaussian quantiles at positions i / (N + 1) for
    # i = 1, ..., N. The positions never reach 0 or 1, so the quantiles are
    # always finite. Computed in float32 for icdf precision, then cast.
    positions = torch.arange(
        start=1, end=num_samples + 1, device=z.device, dtype=torch.float32
    ) / (num_samples + 1)
    quantiles: Tensor = Normal(loc=0.0, scale=1.0).icdf(positions)
    quantiles = quantiles.to(dtype=z.dtype).unsqueeze(-1)

    projections_sorted = projections.sort(dim=-2).values
    loss: Tensor = (projections_sorted - quantiles).square().mean()
    return loss


class VISRegLossComponents(NamedTuple):
    """Individual components of the VISReg loss.

    Attributes:
        total: The combined VISReg loss.
        pred: The multi-view invariance (prediction) loss.
        scale: The unweighted scale loss.
        shape: The unweighted shape loss.
        center: The unweighted center loss.
    """

    total: Tensor
    pred: Tensor
    scale: Tensor
    shape: Tensor
    center: Tensor


class VISRegLoss(nn.Module):
    """Implementation of the VISReg loss [0].

    VISReg regularizes the projected embeddings with three decoupled terms:
    a center loss on the batch mean, a scale loss on the per-dimension
    standard deviation, and a sliced-Wasserstein shape loss that sketches
    the embedding distribution toward an isotropic Gaussian. The
    regularization is combined with the LeJEPA multi-view invariance loss:

    ``total = (1 - lambda_param) * pred + lambda_param * reg`` where
    ``reg = lambda_scale * scale + lambda_shape * shape + lambda_center *
    center``.

    The implementation follows Algorithm 1 of the paper. The default
    ``lambda_param=0.9`` and equal component weights are the best settings
    reported for large datasets such as ImageNet-1K (Tables 3 and 12 in
    [0]); the paper recommends ``lambda_param=0.6`` for small datasets and
    increasing ``lambda_shape`` for long-tailed or low-rank data.

    In distributed training every device draws its own random slices and
    computes the loss on its local batch, so no communication is required
    and the effective number of slices grows with the world size (Section
    3.2 in [0]). This is the paper's setting and the default. Note that
    this requires the random number generators of the devices to be seeded
    differently; identically seeded devices draw identical slices. Set
    ``gather_distributed=True`` to additionally gather the embeddings from
    all devices before computing the regularization statistics, which can
    help when the per-device batch is very small.

    - [0]: VISReg, Wu et al., 2026, https://arxiv.org/abs/2606.02572
    - [1]: https://haiyuwu.github.io/visreg

    Attributes:
        lambda_param: Weight for the regularization term in [0, 1]. The
            invariance term is weighted by (1 - lambda_param).
        num_slices: Number of random 1D projection directions (slices) drawn
            per forward pass on each device.
        lambda_scale: Weight for the scale loss inside the regularization term.
        lambda_shape: Weight for the shape loss inside the regularization term.
        lambda_center: Weight for the center loss inside the regularization
            term.
        gather_distributed: If True, the embeddings from all devices are
            gathered before computing the regularization statistics.
        eps: Numerical stability term for the shape loss normalization.

    Examples:
        >>> # initialize loss function
        >>> loss_fn = VISRegLoss()
        >>>
        >>> # generate local and global views
        >>> local_views = [transform(images) for _ in range(n_local)]
        >>> global_views = [transform(images) for _ in range(n_global)]
        >>>
        >>> # project each view and stack to shape (V, N, D)
        >>> local_proj = torch.stack([model(v) for v in local_views])
        >>> global_proj = torch.stack([model(v) for v in global_views])
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(local_proj=local_proj, global_proj=global_proj)
    """

    def __init__(
        self,
        lambda_param: float = 0.9,
        num_slices: int = 4096,
        lambda_scale: float = 1.0,
        lambda_shape: float = 1.0,
        lambda_center: float = 1.0,
        gather_distributed: bool = False,
        eps: float = 1e-4,
    ):
        """Initializes the VISRegLoss module with the specified parameters.

        Args:
            lambda_param: Weight for the regularization term in ``[0, 1]``.
                The invariance term is weighted by ``1 - lambda_param``.
            num_slices: Number of random 1D projection directions (slices)
                drawn per forward pass on each device.
            lambda_scale: Weight for the scale loss inside the regularization
                term.
            lambda_shape: Weight for the shape loss inside the regularization
                term.
            lambda_center: Weight for the center loss inside the
                regularization term.
            gather_distributed: If True, the embeddings from all devices are
                gathered before computing the regularization statistics.
            eps: Numerical stability term for the shape loss normalization.

        Raises:
            ValueError: If any parameter is outside its valid range or if
                gather_distributed is True but torch.distributed is not
                available.
        """
        super().__init__()
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError("lambda_param must be in the unit interval [0, 1].")
        if num_slices < 1:
            raise ValueError("num_slices must be a positive integer.")
        if lambda_scale < 0 or lambda_shape < 0 or lambda_center < 0:
            raise ValueError(
                "lambda_scale, lambda_shape, and lambda_center must be non-negative."
            )
        if eps <= 0:
            raise ValueError("eps must be greater than zero.")
        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

        self.lambda_param = lambda_param
        self.num_slices = num_slices
        self.lambda_scale = lambda_scale
        self.lambda_shape = lambda_shape
        self.lambda_center = lambda_center
        self.gather_distributed = gather_distributed
        self.eps = eps

    def forward(self, *, local_proj: Tensor, global_proj: Tensor) -> Tensor:
        """Computes the VISReg loss for a batch of multi-view projections.

        Args:
            local_proj: Local-view projected embeddings of shape
                ``(Vl, N, D)`` where ``Vl`` is the number of local views,
                ``N`` is the batch size, and ``D`` is the projection
                dimensionality.
            global_proj: Global-view projected embeddings of shape
                ``(Vg, N, D)`` where ``Vg`` is the number of global views.

        Returns:
            The combined VISReg loss.
        """
        return self.forward_components(
            local_proj=local_proj, global_proj=global_proj
        ).total

    def forward_components(
        self, *, local_proj: Tensor, global_proj: Tensor
    ) -> VISRegLossComponents:
        """Computes the VISReg loss and returns all components separately.

        The prediction term pulls every view, global and local, toward the
        centroid of the global views (Eq. 8 in the paper). The
        regularization terms are computed over all views with each view's
        batch treated as an independent sample set.

        Args:
            local_proj: Local-view projected embeddings of shape
                ``(Vl, N, D)`` where ``Vl`` is the number of local views,
                ``N`` is the batch size, and ``D`` is the projection
                dimensionality.
            global_proj: Global-view projected embeddings of shape
                ``(Vg, N, D)`` where ``Vg`` is the number of global views.

        Returns:
            A VISRegLossComponents tuple with the total loss and the
            individual pred, scale, shape, and center losses.

        Raises:
            ValueError: If the projections have invalid shapes or a batch
                size smaller than two.
        """
        if local_proj.ndim != 3:
            raise ValueError(
                f"local_proj must have shape (V_local, N, D), got {local_proj.shape}."
            )
        if global_proj.ndim != 3:
            raise ValueError(
                f"global_proj must have shape (V_global, N, D), got "
                f"{global_proj.shape}."
            )
        if local_proj.shape[1:] != global_proj.shape[1:]:
            raise ValueError(
                "local_proj and global_proj must have matching batch and feature "
                f"dimensions, got {local_proj.shape} and {global_proj.shape}."
            )
        if local_proj.size(-2) < 2:
            raise ValueError(
                f"Batch size must be greater than one, got {local_proj.size(-2)}."
            )

        proj = torch.cat([global_proj, local_proj], dim=0)
        # Eq. 8 pulls every view, global and local, toward the centroid of
        # the global views.
        pred_loss = lejepa_loss.lejepa_invariance_loss(
            local_proj=proj, global_proj=global_proj
        )

        # The prediction loss above is computed per device (never gathered),
        # matching VICRegLoss. When gathering is enabled, only the
        # regularization statistics below use the batch from all devices.
        if self.gather_distributed and lightly_dist.world_size() > 1:
            proj = torch.cat(lightly_dist.gather(proj), dim=-2)

        scale_loss = visreg_scale_loss(proj)
        shape_loss = visreg_shape_loss(proj, num_slices=self.num_slices, eps=self.eps)
        center_loss = visreg_center_loss(proj)
        reg_loss = (
            self.lambda_scale * scale_loss
            + self.lambda_shape * shape_loss
            + self.lambda_center * center_loss
        )
        total_loss = (
            1.0 - self.lambda_param
        ) * pred_loss + self.lambda_param * reg_loss
        return VISRegLossComponents(
            total=total_loss,
            pred=pred_loss,
            scale=scale_loss,
            shape=shape_loss,
            center=center_loss,
        )
