from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class FroSSLLoss(Module):
    """Implementation of the FroSSL loss [0].

    FroSSL is a self-supervised objective that combines a variance/redundancy
    regularization term with an invariance term. For every view it maximizes the
    entropy of the (trace-normalized) covariance matrix through its squared
    Frobenius norm, while an invariance term pulls the views towards their mean.
    The loss supports any number of views ``V >= 2``.

    - [0] FroSSL: Frobenius Norm Minimization for Self-Supervised Learning, 2024,
        https://arxiv.org/abs/2310.02903

    Attributes:
        invariance_weight:
            Scaling coefficient for the invariance term of the loss.

    Examples:
        >>> # initialize loss function
        >>> loss_fn = FroSSLLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through model
        >>> z0 = model(t0)
        >>> z1 = model(t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn([z0, z1])
    """

    def __init__(self, invariance_weight: float = 1.0) -> None:
        """Initializes the FroSSLLoss module.

        Args:
            invariance_weight:
                Scaling coefficient for the invariance term of the loss.
        """
        super().__init__()
        self.invariance_weight = invariance_weight

    def forward(self, z_views: List[Tensor]) -> Tensor:
        """Computes the FroSSL loss from a list of projected views.

        Args:
            z_views:
                List of ``V`` tensors with shape ``(batch_size, dim)``, one per view.

        Returns:
            The computed FroSSL loss.

        Raises:
            ValueError: If fewer than two views are provided.
        """
        if len(z_views) < 2:
            raise ValueError(
                f"z_views must contain at least two views but found {len(z_views)}."
            )

        num_views = len(z_views)
        dim = z_views[0].size(1)

        # Normalize each view along the batch dimension.
        normalized_z = [F.normalize(z, p=2, dim=0) for z in z_views]
        average_embedding = torch.mean(torch.stack(normalized_z), dim=0)

        total_regularization = z_views[0].new_zeros(())
        total_invariance = z_views[0].new_zeros(())
        for z in normalized_z:
            # Regularization term: maximize the entropy of the (auto)covariance
            # matrix, estimated through its squared Frobenius norm. The smaller
            # of the DxD or NxN matrix is used for efficiency; both share the
            # same non-zero eigenvalues and hence the same Frobenius norm.
            if z.size(0) > dim:
                cov = z.T @ z
            else:
                cov = z @ z.T
            # ``.item()`` matches the reference implementation and avoids a
            # float16 casting issue when dividing by the trace.
            cov = cov / torch.trace(cov).item()
            fro_norm = torch.linalg.norm(cov, ord="fro")
            # The factor of 2 pulls the Frobenius square outside of the log.
            total_regularization = total_regularization + 2 * torch.log(fro_norm)

            # Invariance term: pull each view towards the mean view.
            total_invariance = total_invariance + num_views * dim * F.mse_loss(
                z, average_embedding
            )

        return total_regularization + self.invariance_weight * total_invariance
