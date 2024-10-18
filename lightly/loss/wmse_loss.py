"""Code for W-MSE Loss, largely taken from https://github.com/htdt/self-supervised"""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.linalg.solve_triangular
except ImportError:
    # Only available in PyTorch >=1.11.
    _SOLVE_TRIANGULAR_AVAILABLE = False
else:
    _SOLVE_TRIANGULAR_AVAILABLE = True


def norm_mse_loss(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Normalized MSE Loss as implemented in https://github.com/htdt/self-supervised.

    Args:
        x0: First input tensor.
        x1: Second input tensor.

    Returns:
        The computed normalized MSE loss.
    """
    x0 = F.normalize(x0)
    x1 = F.normalize(x1)
    return torch.sub(input=2, other=(x0 * x1).sum(dim=-1).mean(), alpha=2)


class Whitening2d(nn.Module):
    """Implementation of the whitening layer as described in [0].

    - [0] W-MSE, 2021, https://arxiv.org/pdf/2007.06346.pdf
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.01,
        track_running_stats: bool = True,
        eps: float = 0,
    ):
        """Initializes the Whitening2d module with the specified parameters.

        Args:
            num_features:
                Number of features in the input.
            momentum:
                Momentum for the running mean and variance.
            track_running_stats:
                If True, tracks the running mean and variance.
            eps:
                Epsilon for numerical stability.

        Raises:
            RuntimeError: If torch.linalg.solve_triangular is not available in the PyTorch installation.
        """

        super(Whitening2d, self).__init__()

        if not _SOLVE_TRIANGULAR_AVAILABLE:
            raise RuntimeError(
                "Whitening2d depends on torch.linalg.solve_triangular which is not "
                "available in your PyTorch installation. Please update to PyTorch 1.11 "
                "or newer."
            )

        self.running_mean: torch.Tensor
        self.running_variance: torch.Tensor
        self.num_features = num_features
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.eps = eps

        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros([1, self.num_features, 1, 1])
            )
            self.register_buffer("running_variance", torch.eye(self.num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Whitening2d layer.

        Args:
            x: Input tensor.

        Returns:
            Decorrelated output tensor.

        """
        x = x.unsqueeze(2).unsqueeze(3)
        m = x.mean(0).view(self.num_features, -1).mean(-1).view(1, -1, 1, 1)
        if not self.training and self.track_running_stats:  # for inference
            m = self.running_mean
        xn = x - m

        # Reshape for covariance computation
        T = xn.permute(1, 0, 2, 3).contiguous().view(self.num_features, -1)

        # Compute covariance matrix
        f_cov = torch.mm(T, T.permute(1, 0)) / (T.shape[-1] - 1)

        eye = torch.eye(self.num_features).type(f_cov.type())

        if not self.training and self.track_running_stats:  # for inference
            f_cov = self.running_variance

        f_cov_shrinked = (1 - self.eps) * f_cov + self.eps * eye

        inv_sqrt = torch.linalg.solve_triangular(
            torch.linalg.cholesky(f_cov_shrinked), eye, upper=False
        )

        inv_sqrt = inv_sqrt.contiguous().view(
            self.num_features, self.num_features, 1, 1
        )

        # Decorrelate the features
        decorrelated = F.conv2d(xn, inv_sqrt)

        if self.training and self.track_running_stats:
            self.running_mean = torch.add(
                self.momentum * m.detach(),
                (1 - self.momentum) * self.running_mean,
                out=self.running_mean,
            )
            self.running_variance = torch.add(
                self.momentum * f_cov.detach(),
                (1 - self.momentum) * self.running_variance,
                out=self.running_variance,
            )

        return decorrelated.squeeze(2).squeeze(2)


class WMSELoss(torch.nn.Module):
    """Implementation of the loss described in 'Whitening for
    Self-Supervised Representation Learning' [0].

    - [0] W-MSE, 2021, https://arxiv.org/pdf/2007.06346.pdf

    Examples:
        >>> # initialize loss function
        >>> loss_fn = WMSELoss(num_samples=2)
        >>> transform_fn = WMSETransform(num_samples=2)
        >>>
        >>> # generate the transformed samples
        >>> samples = transform_fn(image)
        >>>
        >>> # feed through encoder head
        >>> h = torch.cat([model(s) for s in samples])
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(samples, h)

    """

    def __init__(
        self,
        embedding_dim: int = 128,
        momentum: float = 0.01,
        eps: float = 0.0,
        track_running_stats: bool = True,
        w_iter: int = 1,
        w_size: int = 256,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = norm_mse_loss,
        num_samples: int = 2,
    ):
        """Initializes the WMSELoss module with the specified parameters.

        Parameters as described in [0].

        Args:
            embedding_dim:
                Dimensionality of the embedding.
            momentum:
                Momentum for the running statistics.
            eps:
                Epsilon for the running statistics.
            track_running_stats:
                Whether to track running statistics.
            w_iter:
                Number of iterations for the whitening.
            w_size:
                Sub-batch size to use for whitening.
            loss_fn:
                Loss function to use for the whitening.
            num_samples:
                Number of samples generated by the transforms for each image.

        Raises:
            ValueError: If w_size is less than twice the size of embedding_dim.
        """
        super().__init__()
        self.whitening = Whitening2d(
            num_features=embedding_dim,
            momentum=momentum,
            eps=eps,
            track_running_stats=track_running_stats,
        )
        if embedding_dim * 2 > w_size:
            raise ValueError(
                "w_size should be at least twice the size of embedding_dim to avoid instabiliy"
            )
        self.w_iter = w_iter
        self.w_size = w_size
        self.loss_f = loss_fn
        self.num_samples = num_samples
        self.num_pairs = num_samples * (num_samples - 1) // 2

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Calculates the W-MSE loss.

        Args:
            input: Tensor with shape (batch_size * num_samples, embedding_dim).

        Returns:
            Aggregate W-MSE loss over all sub-batches.

        Raises:
            RuntimeError: If the batch size is not divisible by num_samples.
            ValueError: If the batch size is smaller than w_size.
        """
        if input.shape[0] % self.num_samples != 0:
            raise RuntimeError("input batch size must be divisible by num_samples")

        bs = input.shape[0] // self.num_samples

        if bs < self.w_size:
            raise ValueError("batch size must be greater than or equal to w_size")
        loss = torch.tensor(0.0, device=input.device, requires_grad=True)

        for _ in range(self.w_iter):
            z = torch.empty_like(input)
            perm = torch.randperm(bs).view(-1, self.w_size)
            for idx in perm:
                for i in range(self.num_samples):
                    z[idx + i * bs] = self.whitening(input[idx + i * bs])
            for i in range(self.num_samples - 1):
                for j in range(i + 1, self.num_samples):
                    x0 = z[i * bs : (i + 1) * bs]
                    x1 = z[j * bs : (j + 1) * bs]
                    loss = loss + self.loss_f(x0, x1)
        loss = loss / (self.w_iter * self.num_pairs)
        return loss
