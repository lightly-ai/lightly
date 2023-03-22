from typing import Callable

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
        super().__init__(
            temperature=temperature,
            sinkhorn_iterations=sinkhorn_iterations,
            regularization_weight=regularization_weight,
            target_distribution="power_law",
            power_law_exponent=power_law_exponent,
            gather_distributed=gather_distributed,
        )


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
        super().__init__(
            temperature=temperature,
            sinkhorn_iterations=sinkhorn_iterations,
            regularization_weight=regularization_weight,
            target_distribution=target_distribution,
            gather_distributed=gather_distributed,
        )
