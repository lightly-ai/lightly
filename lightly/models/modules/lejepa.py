import torch
from torch import nn


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def _generate_unit_vectors(
        self, device: torch.device, num_features: int, num_vectors: int = 256
    ) -> torch.Tensor:
        A = torch.randn(num_features, num_vectors, device=device)
        A = A.div_(A.norm(p=2, dim=0))
        return A

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        proj: (B, N, C)
        """

        num_features = proj.size(-1)
        A = self._generate_unit_vectors(proj.device, num_features)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
