"""LeJEPA loss functions and regularizers."""

import torch
from torch import Tensor, nn
from torch import distributed as torch_dist
from torch.distributed import nn as torch_dist_nn

from lightly.utils import dist as lightly_dist


def lejepa_invariance_loss(*, local_proj: Tensor, global_proj: Tensor) -> Tensor:
    """LeJEPA invariance loss across multiple views.

    Pulls each local view's projection toward the global mean across views.
    Given local projections of shape ``(Vl, N, D)`` and global projections of
    shape ``(Vg, N, D)``, this is the mean-squared distance between every local
    view and the centroid of the global views.

    Reference:
        LeJEPA, 2025, https://arxiv.org/abs/2511.08544

    Args:
        local_proj:
            Projected embeddings of shape ``(Vl, N, D)`` where ``Vl`` is the
            number of local views, ``N`` is the batch size, and ``D`` is the
            projection dimensionality.
        global_proj:
            Projected embeddings of shape ``(Vg, N, D)`` where ``Vg`` is the
            number of global views, ``N`` is the batch size, and ``D`` is the
            projection dimensionality.

    Returns:
        Scalar invariance loss.
    """
    _validate_projection_shapes(local_proj=local_proj, global_proj=global_proj)
    centers = global_proj.mean(0)
    return (centers - local_proj).square().mean()


def _validate_projection_shapes(*, local_proj: Tensor, global_proj: Tensor) -> None:
    if local_proj.ndim != 3:
        raise ValueError(
            f"local_proj must have shape (V_local, N, D), got {local_proj.shape}."
        )
    if global_proj.ndim != 3:
        raise ValueError(
            f"global_proj must have shape (V_global, N, D), got {global_proj.shape}."
        )
    if local_proj.shape[1:] != global_proj.shape[1:]:
        raise ValueError(
            "local_proj and global_proj must have matching batch and feature "
            f"dimensions, got {local_proj.shape} and {global_proj.shape}."
        )
    if local_proj.shape[0] < 1:
        raise ValueError(
            f"local_proj must have at least one local view, got {local_proj.shape}."
        )
    if global_proj.shape[0] < 1:
        raise ValueError(
            f"global_proj must have at least one global view, got {global_proj.shape}."
        )


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization for projected embeddings."""

    def __init__(
        self,
        *,
        knots: int = 17,
        t_max: float = 3.0,
        num_vectors: int = 1024,
        gather_distributed: bool = False,
    ):
        """Initialize the frequency grid and trapezoidal weights.

        All arguments are keyword-only.

        `t_max` sets how far the frequency grid extends. Higher values make the
        loss more sensitive to fine-scale differences in the projected
        distribution, but can also increase noise. `num_vectors` sets how many
        random projection directions are averaged per forward pass. More vectors
        usually improve stability at the cost of extra compute. The defaults
        (`t_max=3.0`, `num_vectors=1024`) follow LeJEPA settings.

        Args:
            knots: Number of frequency samples used for the integration grid.
            t_max: Maximum frequency for the integration grid.
            num_vectors: Number of random unit projection vectors (slices).
            gather_distributed:
                If True, aggregate statistics across distributed ranks so the
                loss uses the global batch.
        """
        super().__init__()
        if knots <= 1:
            raise ValueError("knots must be an integer greater than one.")
        if t_max <= 0:
            raise ValueError("t_max must be greater than zero.")
        if num_vectors <= 0:
            raise ValueError("num_vectors must be a positive integer.")
        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

        self.num_vectors = num_vectors
        self.gather_distributed = gather_distributed

        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        # t are frequencies
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        # phi is the ideal gaussian
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def _generate_unit_vectors(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_features: int,
    ) -> Tensor:
        """Sample unit vectors to project embeddings onto random directions."""
        A = torch.randn(num_features, self.num_vectors, device=device, dtype=dtype)
        if self.gather_distributed and lightly_dist.world_size() > 1:
            torch_dist.broadcast(A, src=0)
        A = A.div_(A.norm(p=2, dim=0))
        return A

    def _project_embeddings_to_unit_vector(
        self,
        proj: Tensor,
        A: Tensor,
    ) -> Tensor:
        """Project embeddings onto the sampled unit vectors."""
        return proj @ A

    def _compute_cf_error_at_each_frequency(
        self,
        x_t: Tensor,
        num_samples: int,
    ) -> Tensor:
        """Compute characteristic function error per frequency."""
        cos_sum = x_t.cos().sum(-3)
        sin_sum = x_t.sin().sum(-3)
        if self.gather_distributed and lightly_dist.world_size() > 1:
            # Using the autograd-aware torch.distributed.nn.all_reduce, ref #1920
            cos_sum = torch_dist_nn.all_reduce(cos_sum)
            sin_sum = torch_dist_nn.all_reduce(sin_sum)

        cos_mean = cos_sum / num_samples
        sin_mean = sin_sum / num_samples
        phi = self.phi.to(dtype=x_t.dtype)
        return (cos_mean - phi).square() + sin_mean.square()  # type: ignore[operator]

    def _integrate_via_trapezoidal_rule(
        self,
        err_per_frequency: Tensor,
        num_samples: int,
    ) -> Tensor:
        """Integrate the error over frequency using trapezoidal weights."""
        weights = self.weights.to(dtype=err_per_frequency.dtype)
        statistic = (err_per_frequency @ weights) * num_samples  # type: ignore[operator]
        return statistic.mean()

    def forward(self, proj: Tensor) -> Tensor:
        """Compute the SIGReg loss for a batch of projections.

        Args:
            proj: Projected embeddings of shape (..., N, C).
        """
        num_features = proj.size(-1)
        num_samples = proj.size(-2)
        if self.gather_distributed and lightly_dist.world_size() > 1:
            num_samples_tensor = torch.tensor(
                float(num_samples), device=proj.device, dtype=proj.dtype
            )
            torch_dist.all_reduce(num_samples_tensor)
            num_samples = int(num_samples_tensor.item())

        A = self._generate_unit_vectors(proj.device, proj.dtype, num_features)
        x_projected = self._project_embeddings_to_unit_vector(proj, A)
        x_t = x_projected.unsqueeze(-1) * self.t  # type: ignore[operator]
        err_per_frequency = self._compute_cf_error_at_each_frequency(x_t, num_samples)
        return self._integrate_via_trapezoidal_rule(err_per_frequency, num_samples)


class LeJEPALoss(nn.Module):
    """LeJEPA loss combining SIGReg regularization with view invariance.

    The loss is a convex combination of two terms:

    - ``SIGReg(local_proj)`` regularizes local projections toward an
      isotropic Gaussian distribution.
    - ``lejepa_invariance_loss(local_proj=local_proj, global_proj=global_proj)``
      pulls each local view toward the mean of the global views.

    The total loss is
    ``lambda_param * SIGReg(local_proj) + (1 - lambda_param) * invariance(local_proj, global_proj)``.

    The default ``lambda_param=0.05`` matches the reference implementation
    [1]. The paper [0] explores values between 0.01 and 0.1.

    - [0]: LeJEPA, 2025, https://arxiv.org/abs/2511.08544
    - [1]: https://github.com/galilai-group/lejepa

    Attributes:
        lambda_param:
            Weight for the SIGReg term in [0, 1]. The invariance term is
            weighted by (1 - lambda_param).
        sigreg:
            The SIGReg regularization module instantiated with the supplied
            sigreg_* parameters.

    Examples:
        >>> # initialize loss function
        >>> loss_fn = LeJEPALoss()
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
        lambda_param: float = 0.05,
        gather_distributed: bool = False,
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        sigreg_num_vectors: int = 1024,
    ):
        """Initialize the combined LeJEPA loss.

        Args:
            lambda_param:
                Weight for the SIGReg term in ``[0, 1]``. The invariance
                term is weighted by ``1 - lambda_param``.
            gather_distributed:
                If True, aggregate SIGReg statistics across distributed
                ranks so the loss uses the global batch.
            sigreg_knots:
                Number of frequency samples used for the SIGReg integration
                grid.
            sigreg_t_max:
                Maximum frequency for the SIGReg integration grid.
            sigreg_num_vectors:
                Number of random unit projection vectors (slices) used by
                SIGReg per forward pass.
        """
        super().__init__()
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError("lambda_param must be in the unit interval [0, 1].")

        self.lambda_param = lambda_param
        self.sigreg = SIGReg(
            knots=sigreg_knots,
            t_max=sigreg_t_max,
            num_vectors=sigreg_num_vectors,
            gather_distributed=gather_distributed,
        )

    def forward(self, *, local_proj: Tensor, global_proj: Tensor) -> Tensor:
        """Compute the LeJEPA loss for a batch of multi-view projections.

        Args:
            local_proj: Local-view projected embeddings of shape ``(Vl, N, D)``.
            global_proj: Global-view projected embeddings of shape ``(Vg, N, D)``.
        """
        _validate_projection_shapes(local_proj=local_proj, global_proj=global_proj)
        sigreg_loss = self.sigreg(local_proj)
        inv_loss = lejepa_invariance_loss(
            local_proj=local_proj, global_proj=global_proj
        )
        loss: Tensor = (
            self.lambda_param * sigreg_loss + (1.0 - self.lambda_param) * inv_loss
        )
        return loss
