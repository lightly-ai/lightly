from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from lightly.loss.msn_loss import MSNLoss


class PMSNLoss(MSNLoss):
    """Implementation of the loss function from PMSN [0] using a power law target
    distribution.

    - [0]: Prior Matching for Siamese Networks, 2022, https://arxiv.org/abs/2210.07277

    Attributes:
        temperature:
            Similarities between anchors and targets are scaled by the inverse of
            the temperature. Must be in (0, inf).
        sinkhorn_iterations:
            Number of sinkhorn normalization iterations on the targets.
        regularization_weight:
            Weight factor lambda by which the regularization loss is scaled. Set to 0
            to disable regularization.
        power_law_exponent:
            Exponent for power law distribution. Entry k of the distribution is
            proportional to (1 / k) ^ power_law_exponent, with k ranging from 1 to dim + 1.
        gather_distributed:
            If True, then target probabilities are gathered from all GPUs.

    Examples:
        >>> # initialize loss function
        >>> loss_fn = PMSNLoss()
        >>>
        >>> # generate anchors and targets of images
        >>> anchors = transforms(images)
        >>> targets = transforms(images)
        >>>
        >>> # feed through PMSN model
        >>> anchors_out = model(anchors)
        >>> targets_out = model.target(targets)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(anchors_out, targets_out, prototypes=model.prototypes)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        regularization_weight: float = 1,
        power_law_exponent: float = 0.25,
        gather_distributed: bool = False,
    ):
        """Initializes the PMSNLoss module with the specified parameters."""
        super().__init__(
            temperature=temperature,
            sinkhorn_iterations=sinkhorn_iterations,
            regularization_weight=regularization_weight,
            gather_distributed=gather_distributed,
        )
        self.power_law_exponent = power_law_exponent

    def regularization_loss(self, mean_anchor_probs: Tensor) -> Tensor:
        """Calculates the regularization loss with a power law target distribution.

        Args:
            mean_anchor_probs: The mean anchor probabilities.

        Returns:
            The calculated regularization loss.
        """
        power_dist = _power_law_distribution(
            size=mean_anchor_probs.shape[0],
            exponent=self.power_law_exponent,
            device=mean_anchor_probs.device,
        )
        loss = F.kl_div(
            input=mean_anchor_probs.log(), target=power_dist, reduction="sum"
        )
        return loss


class PMSNCustomLoss(MSNLoss):
    """Implementation of the loss function from PMSN [0] with a custom target
    distribution.

    - [0]: Prior Matching for Siamese Networks, 2022, https://arxiv.org/abs/2210.07277

    Attributes:
        target_distribution:
            A function that takes the mean anchor probabilities tensor with shape (dim,)
            as input and returns a target probability distribution tensor with the same
            shape. The returned distribution should sum up to one. The final
            regularization loss is calculated as KL(mean_anchor_probs, target_dist)
            where KL is the Kullback-Leibler divergence.
        temperature:
            Similarities between anchors and targets are scaled by the inverse of
            the temperature. Must be in (0, inf).
        sinkhorn_iterations:
            Number of sinkhorn normalization iterations on the targets.
        regularization_weight:
            Weight factor lambda by which the regularization loss is scaled. Set to 0
            to disable regularization.
        gather_distributed:
            If True, then target probabilities are gathered from all GPUs.

    Examples:
        >>> # define custom target distribution
        >>> def my_uniform_distribution(mean_anchor_probabilities: Tensor) -> Tensor:
        >>>     dim = mean_anchor_probabilities.shape[0]
        >>>     return mean_anchor_probabilities.new_ones(dim) / dim
        >>>
        >>> # initialize loss function
        >>> loss_fn = PMSNCustomLoss(target_distribution=my_uniform_distribution)
        >>>
        >>> # generate anchors and targets of images
        >>> anchors = transforms(images)
        >>> targets = transforms(images)
        >>>
        >>> # feed through PMSN model
        >>> anchors_out = model(anchors)
        >>> targets_out = model.target(targets)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(anchors_out, targets_out, prototypes=model.prototypes)
    """

    def __init__(
        self,
        target_distribution: Callable[[Tensor], Tensor],
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        regularization_weight: float = 1,
        gather_distributed: bool = False,
    ):
        """Initializes the PMSNCustomLoss module with the specified parameters."""
        super().__init__(
            temperature=temperature,
            sinkhorn_iterations=sinkhorn_iterations,
            regularization_weight=regularization_weight,
            gather_distributed=gather_distributed,
        )
        self.target_distribution = target_distribution

    def regularization_loss(self, mean_anchor_probs: Tensor) -> Tensor:
        """Calculates regularization loss with a custom target distribution.

        Args:
            mean_anchor_probs:
                The mean anchor probabilities.

        Returns:
            The calculated regularization loss.
        """
        target_dist = self.target_distribution(mean_anchor_probs).to(
            mean_anchor_probs.device
        )
        loss = F.kl_div(
            input=mean_anchor_probs.log(), target=target_dist, reduction="sum"
        )
        return loss


def _power_law_distribution(size: int, exponent: float, device: torch.device) -> Tensor:
    """Returns a power law distribution summing up to 1.

    Args:
        size:
            The size of the distribution.
        exponent:
            The exponent for the power law distribution.
        device:
            The device to create tensor on.

    Returns:
        A power law distribution tensor summing up to 1.
    """
    k = torch.arange(1, size + 1, device=device)
    power_dist = torch.tensor(k ** (-exponent))
    power_dist = power_dist / power_dist.sum()
    return power_dist
