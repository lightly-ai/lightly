"""Code for EMP-SSL Loss, largely taken from https://github.com/tsb0601/EMP-SSL"""

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


def tcr_loss(z: Tensor, eps: float) -> Tensor:
    """Computes the Total Coding Rate (TCR) loss.

    Args:
        z:
            Patch embeddings.
        eps:
            Epsilon value for numerical stability.

    Returns:
        TCR loss.
    """
    _, batch_size, dim = z.shape
    diag = torch.eye(dim, device=z.device).unsqueeze(0)
    # Matrix multiplication over the batch dimension
    einsum = torch.einsum("vbd,vbe->vde", z, z)

    # Calculate the log determinant
    logdet = torch.logdet(diag + dim / (batch_size * eps) * einsum)

    return 0.5 * logdet.mean()


def invariance_loss(z: Tensor) -> Tensor:
    """Calculates the invariance loss, representing the similiarity between the patch embeddings and the average of
    the patch embeddings.

    Args:
        z:
            Patch embeddings.
    Returns:
        Similarity loss.
    """
    # z has shape (num_views, batch_size, dim)

    # Calculate the mean of the patch embeddings across the batch dimension
    z_mean = z.mean(0, keepdim=True)

    return -F.cosine_similarity(z, z_mean, dim=-1).mean()


class EMPSSLLoss(Module):
    """Implementation of the loss from 'EMP-SSL: Towards Self-Supervised Learning in
    One Training Epoch' [0].

    - [0] EMP-SSL, 2023, https://arxiv.org/abs/2304.03977

    Attributes:
        tcr_eps:
            Total Coding Rate (TCR) epsilon. NOTE: While in the paper, this term is
            squared, we do not square it here as to follow the implementation in the
            official repository.
        inv_coef:
            Coefficient for the invariance loss (Lambda in the paper).

    Examples:
        >>> # initialize loss function
        >>> loss_fn = EMP_SSLLoss()
        >>> base_transform = VICRegViewTransform() # As discussed in paper
        >>> transform_fn = MultiCropTransform(transforms=base_transform, crop_counts=100)
        >>>
        >>> # generate the transformed samples
        >>> samples = transform_fn(image)
        >>>
        >>> # feed through encoder head
        >>> z = torch.cat([model(s) for s in samples])
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(z)
    """

    def __init__(
        self,
        tcr_eps: float = 0.2,
        inv_coef: float = 200.0,
    ) -> None:
        """Initializes the EMPSSLoss module.

        Args:
            tcr_eps:
                Total coding rate (TCR) epsilon.
            inv_coff:
                Coefficient for the invariance loss.
        """
        super().__init__()
        self.tcr_eps = tcr_eps
        self.inv_coef = inv_coef

    def forward(self, z_views: List[Tensor]) -> Tensor:
        """Computes the EMP-SSL loss, which is a combination of Total Coding Rate loss and invariance loss.

        Args:
            z_views:
                List of patch embeddings tensors from different views.

        Returns:
            The computed EMP-SSL loss.
        """

        # z has shape (num_views, batch_size, dim)
        z = torch.stack(z_views)

        return tcr_loss(z, eps=self.tcr_eps) + self.inv_coef * invariance_loss(z)
