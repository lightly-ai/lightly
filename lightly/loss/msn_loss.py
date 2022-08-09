import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def prototype_probabilities(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
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

def sharpen(probabilities: torch.Tensor, temperature: float) -> torch.Tensor:
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
    probabilities: torch.Tensor, 
    iterations: int = 3,
) -> torch.Tensor:
    """Runs sinkhorn normalization on the predictions."""
    num_targets, num_prototypes = probabilities.shape
    probabilities = probabilities.T
    probabilities = probabilities / torch.sum(probabilities)

    for _ in range(iterations):
        # normalize rows
        probabilities /= torch.sum(probabilities, dim=1, keepdim=True)
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
            the temperature. Must be in (0, 1].
        sinkhorn_iterations:
            Number of sinkhorn normalization iterations on the targets.
        me_max_weight:
            Weight factor lambda by which the mean entropy maximization regularization
            loss is scaled. Set to 0 to disable the reguliarization.

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
        me_max_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.me_max_weight = me_max_weight
    
    def forward(
        self,
        anchors: torch.Tensor, 
        targets: torch.Tensor, 
        prototypes: torch.Tensor,
        target_sharpen_temperature: float = 0.25,
    ) -> torch.Tensor:
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
        anchor_probs = prototype_probabilities(anchors, prototypes, temperature=self.temperature)
        
        # target predictions
        with torch.no_grad():
            target_probs = prototype_probabilities(targets, prototypes, temperature=self.temperature)
            target_probs = sharpen(target_probs, temperature=target_sharpen_temperature)
            if self.sinkhorn_iterations > 0:
                target_probs = sinkhorn(target_probs, iterations=self.sinkhorn_iterations)
            target_probs = target_probs.repeat((num_views, 1))

        # cross entropy loss
        loss = torch.mean(torch.sum(torch.log(anchor_probs**(-target_probs)), dim=1))

        # mean entropy maximization regularization
        if self.me_max_weight > 0:
            mean_anchor_probs = torch.mean(anchor_probs, dim=0)
            me_max_loss = torch.sum(torch.log(mean_anchor_probs**(-mean_anchor_probs)))
            me_max_loss += math.log(float(len(mean_anchor_probs)))
            loss -= self.me_max_weight * me_max_loss

        return loss
