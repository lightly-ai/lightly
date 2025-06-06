from functools import partial
from typing import Callable, Optional

import torch
from torch import Tensor
from torch import distributed as torch_dist
from torch import nn

from lightly.utils import dist


def negative_mises_fisher_weights(
    out0: Tensor, out1: Tensor, sigma: float = 0.5
) -> Tensor:
    """Negative Mises-Fisher weighting function as presented in Decoupled Contrastive Learning [0].

    The implementation was inspired by [1].

    - [0] Chun-Hsiao Y. et. al., 2021, Decoupled Contrastive Learning https://arxiv.org/abs/2110.06848
    - [1] https://github.com/raminnakhli/Decoupled-Contrastive-Learning

    Args:
        out0:
            Output projections of the first set of transformed images.
            Shape: (batch_size, embedding_size)
        out1:
            Output projections of the second set of transformed images.
            Shape: (batch_size, embedding_size)
        sigma:
            Similarities are scaled by inverse sigma.
    Returns:
        A tensor with shape (batch_size,) where each entry is the weight for one
        of the input images.
    """
    similarity = torch.einsum("nm,nm->n", out0.detach(), out1.detach()) / sigma

    # Return negative Mises-Fisher weights
    return torch.tensor(2 - out0.shape[0] * nn.functional.softmax(similarity, dim=0))


class DCLLoss(nn.Module):
    """Implementation of the Decoupled Contrastive Learning Loss from Decoupled Contrastive Learning [0].

    This code implements Equation 6 in [0], including the sum over all images `i`
    and views `k`. The loss is reduced to a mean loss over the mini-batch.
    The implementation was inspired by [1].

    - [0] Chun-Hsiao Y. et. al., 2021, Decoupled Contrastive Learning https://arxiv.org/abs/2110.06848
    - [1] https://github.com/raminnakhli/Decoupled-Contrastive-Learning

    Attributes:
        temperature:
            Similarities are scaled by inverse temperature.
        weight_fn:
            Weighting function `w` from the paper. Scales the loss between the
            positive views (views from the same image). No weighting is performed
            if weight_fn is None. The function must take the two input tensors
            passed to the forward call as input and return a weight tensor. The
            returned weight tensor must have the same length as the input tensors.
        gather_distributed:
            If True, negatives from all GPUs are gathered before the
            loss calculation.

    Examples:
        >>> loss_fn = DCLLoss(temperature=0.07)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # embed images using some model, for example SimCLR
        >>> out0 = model(t0)
        >>> out1 = model(t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
        >>>
        >>> # you can also add a custom weighting function
        >>> weight_fn = lambda out0, out1: torch.sum((out0 - out1) ** 2, dim=1)
        >>> loss_fn = DCLLoss(weight_fn=weight_fn)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        weight_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        gather_distributed: bool = False,
    ):
        """Initialzes the DCLoss module.

        Args:
            temperature:
                Similarities are scaled by inverse temperature.
            weight_fn:
                 Weighting function `w` from the paper. Scales the loss between the
                positive views (views from the same image). No weighting is performed
                if weight_fn is None. The function must take the two input tensors
                passed to the forward call as input and return a weight tensor. The
                returned weight tensor must have the same length as the input tensors.
            gather_distributed:
                If True, negatives from all GPUs are gathered before the
                loss calculation.

        Raises:
            ValuesError: If gather_distributed is True but torch.distributed is not available.
        """
        super().__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn
        self.gather_distributed = gather_distributed

        # Check if distributed gathering is enabled but torch.distributed is not available
        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(
        self,
        out0: Tensor,
        out1: Tensor,
    ) -> Tensor:
        """Forward pass of the DCL loss.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)

        Returns:
            Mean loss over the mini-batch.
        """
        # Normalize the output to length 1
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        if self.gather_distributed and dist.world_size() > 1:
            # Gather representations from other processes if necessary
            out0_all = torch.cat(dist.gather(out0), 0)
            out1_all = torch.cat(dist.gather(out1), 0)
        else:
            out0_all = out0
            out1_all = out1

        # Calculate symmetric loss
        loss0: Tensor = self._loss(out0, out1, out0_all, out1_all)
        loss1: Tensor = self._loss(out1, out0, out1_all, out0_all)

        # Return the mean loss over the mini-batch
        return 0.5 * (loss0 + loss1)

    def _loss(
        self, out0: Tensor, out1: Tensor, out0_all: Tensor, out1_all: Tensor
    ) -> Tensor:
        """Calculates DCL loss for out0 with respect to its positives in out1
        and the negatives in out1, out0_all, and out1_all.

        This code implements Equation 6 in [0], including the sum over all images `i`
        but with `k` fixed at 0.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)
            out0_all:
                Output projections of the first set of transformed images from
                all distributed processes/gpus. Should be equal to out0 in an
                undistributed setting.
                Shape: (batch_size * world_size, embedding_size)
            out1_all:
                Output projections of the second set of transformed images from
                all distributed processes/GPUs. Should be equal to out1 in an
                undistributed setting.
                Shape: (batch_size * world_size, embedding_size)

        Returns:
            Mean loss over the mini-batch.
        """
        # Create diagonal mask that only selects similarities between
        # representations of the same images
        batch_size = out0.shape[0]
        if self.gather_distributed and dist.world_size() > 1:
            diag_mask = dist.eye_rank(batch_size, device=out0.device)
        else:
            diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)

        # Calculate similarities (n = batch_size, m = batch_size * world_size)
        sim_00 = torch.einsum("nc,mc->nm", out0, out0_all) / self.temperature
        sim_01 = torch.einsum("nc,mc->nm", out0, out1_all) / self.temperature

        positive_loss = -sim_01[diag_mask]
        if self.weight_fn:
            positive_loss = positive_loss * self.weight_fn(out0, out1)

        # Remove simliarities between same views of the same image
        sim_00 = sim_00[~diag_mask].view(batch_size, -1)

        # Remove similarities between different views of the same images
        # This is the key difference compared to NTXentLoss
        sim_01 = sim_01[~diag_mask].view(batch_size, -1)

        all_negs = torch.cat([sim_00, sim_01], dim=1)
        negative_loss = torch.logsumexp(all_negs, dim=1)  # log(sum exp over *all* negs)
        return (positive_loss + negative_loss).mean()


class DCLWLoss(DCLLoss):
    """Implementation of the Weighted Decoupled Contrastive Learning Loss from
    Decoupled Contrastive Learning [0].

    This code implements Equation 6 in [0] with a negative Mises-Fisher
    weighting function. The loss returns the mean over all images `i` and
    views `k` in the mini-batch. The implementation was inspired by [1].

    - [0] Chun-Hsiao Y. et. al., 2021, Decoupled Contrastive Learning https://arxiv.org/abs/2110.06848
    - [1] https://github.com/raminnakhli/Decoupled-Contrastive-Learning

    Attributes:
        temperature:
            Similarities are scaled by inverse temperature.
        sigma:
            Similar to temperature but applies the inverse scaling in the
            weighting function.
        gather_distributed:
            If True, negatives from all GPUs are gathered before the
            loss calculation.

    Examples:
        >>> loss_fn = DCLWLoss(temperature=0.07)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # embed images using some model, for example SimCLR
        >>> out0 = model(t0)
        >>> out1 = model(t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        sigma: float = 0.5,
        gather_distributed: bool = False,
    ):
        """Initializes the DCLWLoss module.

        Args:
            temperature:
                Similarities are scaled by inverse temperature.
            sigma:
                Applies inverse scaling in the weighting function.
            gather_distributed:
                If True, negatives from all GPUs are gathered before the
                loss calculation.
        """
        super().__init__(
            temperature=temperature,
            weight_fn=partial(negative_mises_fisher_weights, sigma=sigma),
            gather_distributed=gather_distributed,
        )
