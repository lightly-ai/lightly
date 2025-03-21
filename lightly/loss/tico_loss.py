from typing import Union

import torch
import torch.distributed as dist
from torch import Tensor

from lightly.utils.dist import gather


class TiCoLoss(torch.nn.Module):
    """Implementation of the Tico Loss from Tico[0] paper.

    This implementation takes inspiration from the code published
    by sayannag using Lightly. [1]

    - [0] Jiachen Zhu et. al, 2022, Tico... https://arxiv.org/abs/2206.10698
    - [1] https://github.com/sayannag/TiCo-pytorch

    Attributes:
        Args:
            beta:
                Coefficient for the EMA update of the covariance
                Defaults to 0.9 [0].
            rho:
                Weight for the covariance term of the loss
                Defaults to 8.0 [0].
            gather_distributed:
                If True, the cross-correlation matrices from all GPUs are
                gathered and summed before the loss calculation.

    Examples:
        >>> # initialize loss function
        >>> loss_fn = TiCoLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
    """

    def __init__(
        self,
        beta: float = 0.9,
        rho: float = 8.0,
        gather_distributed: bool = False,
    ):
        """Initializes the TiCoLoss module with the specified parameters.

        Args:
            beta:
                Coefficient for the EMA update of the covariance.
            rho:
                Weight for the covariance term of the loss.
            gather_distributed:
                If True, the cross-correlation matrices from all GPUs are gathered
                    and summed before the loss calculation. Default is False.

        Raises:
            ValueError: If gather_distributed is True but torch.distributed is not available.
        """
        super(TiCoLoss, self).__init__()
        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

        self.beta = beta
        self.rho = rho
        self.C: Union[Tensor, None] = None
        self.gather_distributed = gather_distributed

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        update_covariance_matrix: bool = True,
    ) -> torch.Tensor:
        """Computes the TiCo loss.

        It maximizes the agreement among embeddings of different distorted versions of the same image
        while avoiding collapse using Covariance matrix.

        Args:
            z_a:
                Tensor of shape [batch_size, num_features=256]. Output of the learned backbone.
            z_b:
                Tensor of shape [batch_size, num_features=256]. Output of the momentum updated backbone.
            update_covariance_matrix:
                Parameter to update the covariance matrix at each iteration.

        Returns:
            The computed loss.

        Raises:
            AssertionError: If z_a or z_b have a batch size <= 1.
            AssertionError: If z_a and z_b do not have the same shape.
        """

        assert (
            z_a.shape[0] > 1 and z_b.shape[0] > 1
        ), f"z_a and z_b must have batch size > 1 but found {z_a.shape[0]} and {z_b.shape[0]}"
        assert (
            z_a.shape == z_b.shape
        ), f"z_a and z_b must have same shape but found {z_a.shape} and {z_b.shape}."

        # gather all batches
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                z_a = torch.cat(gather(z_a), dim=0)
                z_b = torch.cat(gather(z_b), dim=0)

        # Normalize image
        z_a = torch.nn.functional.normalize(z_a, dim=1)
        z_b = torch.nn.functional.normalize(z_b, dim=1)

        # Compute auxiliary matrix B
        B = torch.mm(z_a.T, z_a).detach() / z_a.shape[0]

        # Initialize covariance matrix
        if self.C is None:
            self.C = B.new_zeros(B.shape).detach()

        # Compute loss
        C = self.beta * self.C + (1 - self.beta) * B

        transformative_invariance_loss = 1.0 - (z_a * z_b).sum(dim=1).mean()
        covariance_contrast_loss = self.rho * (torch.mm(z_a, C) * z_a).sum(dim=1).mean()

        loss: Tensor = transformative_invariance_loss + covariance_contrast_loss

        # Update covariance matrix
        if update_covariance_matrix:
            self.C = C.detach()

        return loss
