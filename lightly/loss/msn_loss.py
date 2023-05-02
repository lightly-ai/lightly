import math
import warnings
from typing import Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def prototype_probabilities(
    queries: Tensor,
    prototypes: Tensor,
    temperature: float,
) -> Tensor:
    """Returns probability for each query to belong to each prototype.

    Args:
        queries:
            Tensor with shape (batch_size, dim)
        prototypes:
            Tensor with shape (num_prototypes, dim)
        temperature:
            Inverse scaling factor for the similarity.

    Returns:
        Probability tensor with shape (batch_size, num_prototypes) which sums to 1 along
        the num_prototypes dimension.

    """
    return F.softmax(torch.matmul(queries, prototypes.T) / temperature, dim=1)


def sharpen(probabilities: Tensor, temperature: float) -> Tensor:
    """Sharpens the probabilities with the given temperature.

    Args:
        probabilities:
            Tensor with shape (batch_size, dim)
        temperature:
            Temperature in (0, 1]. Lower temperature results in stronger sharpening (
            output probabilities are less uniform).
    Returns:
        Probabilities tensor with shape (batch_size, dim).

    """
    probabilities = probabilities ** (1.0 / temperature)
    probabilities /= torch.sum(probabilities, dim=1, keepdim=True)
    return probabilities


@torch.no_grad()
def sinkhorn(
    probabilities: Tensor,
    iterations: int = 3,
    gather_distributed: bool = False,
) -> Tensor:
    """Runs sinkhorn normalization on the probabilities as described in [0].

    Code inspired by [1].

    - [0]: Masked Siamese Networks, 2022, https://arxiv.org/abs/2204.07141
    - [1]: https://github.com/facebookresearch/msn

    Args:
        probabilities:
            Probabilities tensor with shape (batch_size, num_prototypes).
        iterations:
            Number of iterations of the sinkhorn algorithms. Set to 0 to disable.
        gather_distributed:
            If True then features from all gpus are gathered during normalization.
    Returns:
        A normalized probabilities tensor.

    """
    if iterations <= 0:
        return probabilities

    world_size = 1
    if gather_distributed and dist.is_initialized():
        world_size = dist.get_world_size()

    num_targets, num_prototypes = probabilities.shape
    probabilities = probabilities.T
    sum_probabilities = torch.sum(probabilities)
    if world_size > 1:
        dist.all_reduce(sum_probabilities)
    probabilities = probabilities / sum_probabilities

    for _ in range(iterations):
        # normalize rows
        row_sum = torch.sum(probabilities, dim=1, keepdim=True)
        if world_size > 1:
            dist.all_reduce(row_sum)
        probabilities /= row_sum
        probabilities /= num_prototypes

        # normalize columns
        probabilities /= torch.sum(probabilities, dim=0, keepdim=True)
        probabilities /= num_targets

    probabilities *= num_targets
    return probabilities.T


class MSNLoss(nn.Module):
    """Implementation of the loss function from MSN [0].

    Code inspired by [1].

    - [0]: Masked Siamese Networks, 2022, https://arxiv.org/abs/2204.07141
    - [1]: https://github.com/facebookresearch/msn

    Attributes:
        temperature:
            Similarities between anchors and targets are scaled by the inverse of
            the temperature. Must be in (0, inf).
        sinkhorn_iterations:
            Number of sinkhorn normalization iterations on the targets.
        regularization_weight:
            Weight factor lambda by which the regularization loss is scaled. Set to 0
            to disable regularization.
        me_max_weight:
            Deprecated, use `regularization_weight` instead. Takes precendence over
            `regularization_weight` if not None. Weight factor lambda by which the mean
            entropy maximization regularization loss is scaled. Set to 0 to disable
            mean entropy maximization reguliarization.
        gather_distributed:
            If True, then target probabilities are gathered from all GPUs.

     Examples:

        >>> # initialize loss function
        >>> loss_fn = MSNLoss()
        >>>
        >>> # generate anchors and targets of images
        >>> anchors = transforms(images)
        >>> targets = transforms(images)
        >>>
        >>> # feed through MSN model
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
        regularization_weight: float = 1.0,
        me_max_weight: Union[float, None] = None,
        gather_distributed: bool = False,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be in (0, inf) but is {temperature}.")
        if sinkhorn_iterations < 0:
            raise ValueError(
                f"sinkhorn_iterations must be >= 0 but is {sinkhorn_iterations}."
            )
        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.regularization_weight = regularization_weight
        # set regularization_weight to me_max_weight for backwards compatibility
        if me_max_weight is not None:
            warnings.warn(
                DeprecationWarning(
                    "me_max_weight is deprecated in favor of regularization_weight and "
                    "will be removed in the future."
                )
            )
            self.regularization_weight = me_max_weight
        self.gather_distributed = gather_distributed

    def forward(
        self,
        anchors: Tensor,
        targets: Tensor,
        prototypes: Tensor,
        target_sharpen_temperature: float = 0.25,
    ) -> Tensor:
        """Computes the MSN loss for a set of anchors, targets and prototypes.

        Args:
            anchors:
                Tensor with shape (batch_size * anchor_views, dim).
            targets:
                Tensor with shape (batch_size, dim).
            prototypes:
                Tensor with shape (num_prototypes, dim).
            target_sharpen_temperature:
                Temperature used to sharpen the target probabilities.

        Returns:
            Mean loss over all anchors.

        """
        num_views = anchors.shape[0] // targets.shape[0]
        anchors = F.normalize(anchors, dim=1)
        targets = F.normalize(targets, dim=1)
        prototypes = F.normalize(prototypes, dim=1)

        # anchor predictions
        anchor_probs = prototype_probabilities(
            anchors, prototypes, temperature=self.temperature
        )

        # target predictions
        with torch.no_grad():
            target_probs = prototype_probabilities(
                targets, prototypes, temperature=self.temperature
            )
            target_probs = sharpen(target_probs, temperature=target_sharpen_temperature)
            if self.sinkhorn_iterations > 0:
                target_probs = sinkhorn(
                    probabilities=target_probs,
                    iterations=self.sinkhorn_iterations,
                    gather_distributed=self.gather_distributed,
                )
            target_probs = target_probs.repeat((num_views, 1))

        # cross entropy loss
        loss = torch.mean(torch.sum(torch.log(anchor_probs ** (-target_probs)), dim=1))

        # regularization loss
        if self.regularization_weight > 0:
            mean_anchor_probs = torch.mean(anchor_probs, dim=0)
            reg_loss = self.regularization_loss(mean_anchor_probs=mean_anchor_probs)
            loss += self.regularization_weight * reg_loss

        return loss

    def regularization_loss(self, mean_anchor_probs: Tensor) -> Tensor:
        """Calculates mean entropy regularization loss."""
        loss = -torch.sum(torch.log(mean_anchor_probs ** (-mean_anchor_probs)))
        loss += math.log(float(len(mean_anchor_probs)))
        return loss
