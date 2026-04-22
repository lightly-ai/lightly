import torch
from torch import distributed as torch_dist
from torch import nn

from lightly.utils import dist as lightly_dist


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization for projected embeddings."""

    def __init__(
        self,
        knots: int = 17,
        t_max: float = 3.0,
        num_vectors: int = 256,
        gather_distributed: bool = False,
    ):
        """Initialize the frequency grid and trapezoidal weights.

        `t_max` sets how far the frequency grid extends. Higher values make the
        loss more sensitive to fine-scale differences in the projected
        distribution, but can also increase noise. `num_vectors` sets how many
        random projection directions are averaged per forward pass. More vectors
        usually improve stability at the cost of extra compute. The defaults
        (`t_max=3.0`, `num_vectors=256`) follow LeJEPA settings.

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
    ) -> torch.Tensor:
        """Sample unit vectors to project embeddings onto random directions."""
        A = torch.randn(num_features, self.num_vectors, device=device, dtype=dtype)
        if self.gather_distributed and lightly_dist.world_size() > 1:
            torch_dist.broadcast(A, src=0)
        A = A.div_(A.norm(p=2, dim=0))
        return A

    def _project_embeddings_to_unit_vector(
        self,
        proj: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Project embeddings onto the sampled unit vectors."""
        return proj @ A

    def _compute_cf_error_at_each_frequency(
        self,
        x_t: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Compute characteristic function error per frequency."""
        cos_sum = x_t.cos().sum(-3)
        sin_sum = x_t.sin().sum(-3)
        if self.gather_distributed and lightly_dist.world_size() > 1:
            torch_dist.all_reduce(cos_sum)
            torch_dist.all_reduce(sin_sum)

        cos_mean = cos_sum / num_samples
        sin_mean = sin_sum / num_samples
        phi = self.phi.to(dtype=x_t.dtype)
        return (cos_mean - phi).square() + sin_mean.square()  # type: ignore[operator]

    def _integrate_via_trapezoidal_rule(
        self,
        err_per_frequency: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Integrate the error over frequency using trapezoidal weights."""
        weights = self.weights.to(dtype=err_per_frequency.dtype)
        statistic = (err_per_frequency @ weights) * num_samples  # type: ignore[operator]
        return statistic.mean()

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Compute the SIGReg loss for a batch of projections.

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


def lejepa_invariance_loss(proj: torch.Tensor) -> torch.Tensor:
    """LeJEPA invariance loss across multiple views.

    Pulls each view's projection toward the per-sample mean across views.
    Given projections of shape ``(V, N, D)``, this is the mean-squared
    distance between every view and the per-sample centroid computed over
    the view dimension.

    Reference:
        LeJEPA, 2025, https://arxiv.org/abs/2511.08544

    Args:
        proj:
            Projected embeddings of shape ``(V, N, D)`` where ``V`` is the
            number of views, ``N`` is the batch size, and ``D`` is the
            projection dimensionality.

    Returns:
        Scalar invariance loss.
    """
    return (proj.mean(0) - proj).square().mean()


class LeJEPALoss(nn.Module):
    """LeJEPA loss combining SIGReg regularization with view invariance.

    The loss is a convex combination of two terms:

    - ``SIGReg(proj)`` regularizes projections toward an isotropic Gaussian
      distribution.
    - ``lejepa_invariance_loss(proj)`` pulls each view toward the per-sample
      mean across views.

    The total loss is
    ``lambda_param * SIGReg(proj) + (1 - lambda_param) * invariance(proj)``.

    The default ``lambda_param=0.02`` matches the reference implementation
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
        >>> # generate multiple views of the same images
        >>> views = [transform(images) for _ in range(n_views)]
        >>>
        >>> # project each view and stack to shape (V, N, D)
        >>> proj = torch.stack([model(v) for v in views])
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(proj)
    """

    def __init__(
        self,
        lambda_param: float = 0.02,
        gather_distributed: bool = False,
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        sigreg_num_vectors: int = 256,
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

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute the LeJEPA loss for a batch of multi-view projections.

        Args:
            proj: Projected embeddings of shape ``(V, N, D)``.
        """
        sigreg_loss = self.sigreg(proj)
        inv_loss = lejepa_invariance_loss(proj)
        loss: torch.Tensor = (
            self.lambda_param * sigreg_loss + (1.0 - self.lambda_param) * inv_loss
        )
        return loss
