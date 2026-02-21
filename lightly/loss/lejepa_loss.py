import torch
from torch import nn


class SIGReg(nn.Module):
    """Stochastic integral Gaussian regularizer for projected embeddings."""

    def __init__(self, knots: int = 17):
        """Initialize the frequency grid and trapezoidal weights.

        Args:
            knots: Number of frequency samples used for the integration grid.
        """
        super().__init__()
        if knots <= 1:
            raise ValueError("knots must be an integer greater than one.")

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        # t are frequencies
        dt = 3 / (knots - 1)
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
        num_vectors: int = 256,
    ) -> torch.Tensor:
        """Sample unit vectors to project embeddings onto random directions."""
        A = torch.randn(num_features, num_vectors, device=device, dtype=dtype)
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
    ) -> torch.Tensor:
        """Compute characteristic function error per frequency."""
        phi = self.phi.to(dtype=x_t.dtype)
        return (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()

    def _integrate_via_trapezoidal_rule(
        self,
        err_per_frequency: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Integrate the error over frequency using trapezoidal weights."""
        weights = self.weights.to(dtype=err_per_frequency.dtype)
        statistic = (err_per_frequency @ weights) * num_samples
        return statistic.mean()

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Compute the SIGReg loss for a batch of projections.

        Args:
            proj: Projected embeddings of shape (B, N, C).
        """

        num_features = proj.size(-1)
        num_samples = proj.size(-2)
        A = self._generate_unit_vectors(proj.device, proj.dtype, num_features)
        x_projected = self._project_embeddings_to_unit_vector(proj, A)
        x_t = x_projected.unsqueeze(-1) * self.t
        err_per_frequency = self._compute_cf_error_at_each_frequency(x_t)
        return self._integrate_via_trapezoidal_rule(err_per_frequency, num_samples)
