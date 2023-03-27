from typing import Tuple

import torch
import torch.nn.functional as F

from lightly.loss.vicreg_loss import VICRegLoss
from lightly.models.utils import nearest_neighbors


class VICRegLLoss(torch.nn.Module):
    """Implementation of the VICRegL loss from VICRegL paper [0].

    This implementation follows the code published by the authors. [1]

    - [0]: VICRegL, 2022, https://arxiv.org/abs/2210.01571
    - [1]: https://github.com/facebookresearch/VICRegL

    Attributes:
        lambda_param:
            Coefficient for the invariance term of the loss.
        mu_param:
            Coefficient for the variance term of the loss.
        nu_param:
            Coefficient for the covariance term of the loss.
        alpha:
            Coefficient to weight global with local loss. The final loss is computed as
            (self.alpha * global_loss + (1-self.alpha) * local_loss).
        gather_distributed:
            If True then the cross-correlation matrices from all gpus are gathered and
            summed before the loss calculation.
        eps:
            Numerical epsilon.
        num_matches:
            Number of local matches between patches in the KNN search.

    Examples:

        >>> # initialize loss function
        >>> loss_fn = VICRegLLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through model images
        >>> out0, out1, out_local0, out_local1, grid0, grid1  = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1, out_local0, out_local1, grid0, grid1)
    """

    def __init__(
        self,
        lambda_param: float = 25.0,
        mu_param: float = 25.0,
        nu_param: float = 1.0,
        alpha: float = 0.25,
        gather_distributed: bool = False,
        eps: float = 0.0001,
        num_matches: Tuple[int, int] = (20, 4),
    ):
        super(VICRegLLoss, self).__init__()
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.gather_distributed = gather_distributed
        self.alpha = alpha
        self.eps = eps
        self.num_matches = num_matches
        self.vicregloss = VICRegLoss()

    def _nearest_neighbors_on_l2(
        self, input_maps: torch.Tensor, candidate_maps: torch.Tensor, num_matches: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_maps: (B, H * W, C)
        candidate_maps: (B, H * W, C)
        """
        distances = torch.cdist(input_maps, candidate_maps)
        return nearest_neighbors(input_maps, candidate_maps, distances, num_matches)

    def _nearest_neighbors_on_grid(
        self,
        input_grid: torch.Tensor,
        candidate_grid: torch.Tensor,
        input_maps: torch.Tensor,
        candidate_maps: torch.Tensor,
        num_matches: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_grid: (B, H * W, 2)
        candidate_grid: (B, H * W, 2)
        input_maps: (B, H * W, C)
        candidate_maps: (B, H * W, C)
        """
        distances = torch.cdist(input_grid, candidate_grid)
        return nearest_neighbors(input_maps, candidate_maps, distances, num_matches)

    def local_loss(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        grid_a: torch.Tensor,
        grid_b: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the local loss between two sets of local features using nearest
        neighbors.

        Args:
            z_a:
                A local features tensor from a global view. Must have size:
                (batch_size, global_view_height, global_view_width, global_feature_dim).
            z_b:
                A local features tensor from a global or local view. Must have
                size: (batch_size, height, width, global_feature_dim) where height and
                width depend on the view type.
            grid_a:
                A tensor of grids from a global view. Must have size:
                (batch_size, grid_size, grid_size, 2).
            grid_b:
                A tensor of grids from a global or local  view. Musth have size:
                (batch_size, grid_size, grid_size, 2).

        Returns:
            The local loss.
        """
        inv_loss = 0.0

        z_a = z_a.flatten(1, 2)
        z_b = z_b.flatten(1, 2)

        # L2 loss
        # Check if one of the feature tensors comes from a local view. In this case we
        # reduce the number of matches.
        has_local_view = z_a.shape[1] != z_b.shape[1]
        num_matches = self.num_matches[1] if has_local_view else self.num_matches[0]
        z_a_filtered, z_a_nn = self._nearest_neighbors_on_l2(
            input_maps=z_a, candidate_maps=z_b, num_matches=num_matches
        )
        z_b_filtered, z_b_nn = self._nearest_neighbors_on_l2(
            input_maps=z_b, candidate_maps=z_a, num_matches=num_matches
        )

        inv_loss_a = F.mse_loss(z_a_filtered, z_a_nn)
        inv_loss_b = F.mse_loss(z_b_filtered, z_b_nn)
        inv_loss = inv_loss + (inv_loss_a / 2 + inv_loss_b / 2)

        grid_a = grid_a.flatten(1, 2)
        grid_b = grid_b.flatten(1, 2)

        # distance loss
        z_a_filtered, z_a_nn = self._nearest_neighbors_on_grid(
            input_grid=grid_a,
            candidate_grid=grid_b,
            input_maps=z_a,
            candidate_maps=z_b,
            num_matches=self.num_matches[0],
        )

        z_b_filtered, z_b_nn = self._nearest_neighbors_on_grid(
            input_grid=grid_b,
            candidate_grid=grid_a,
            input_maps=z_b,
            candidate_maps=z_a,
            num_matches=self.num_matches[1],
        )

        inv_loss_a = F.mse_loss(z_a_filtered, z_a_nn)
        inv_loss_b = F.mse_loss(z_b_filtered, z_b_nn)
        inv_loss = inv_loss + (inv_loss_a / 2 + inv_loss_b / 2)

        local_loss = self.lambda_param * inv_loss

        return local_loss

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        z_a_local_features: torch.Tensor,
        z_b_local_features: torch.Tensor,
        grid_a: torch.Tensor,
        grid_b: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the overall loss between two sets of feature maps, using global and
        local loss.

        Computes the global loss using VicRegLoss and local loss using nearest neighbor
        search. The final loss is computed as
        (self.alpha * global_loss + (1-self.alpha) * local_loss).

        Args:
            z_a:
                A global features tensor from a global view. Must have size:
                (batch_size, global_height, global_width, global_feature_dimension)
            z_b:
                A global features tensor from a global or a local view. Must have size:
                (batch_size, height, width, global_feature_dim) where height and width
                depend on the view type.
            z_a_local_features:
                A local features tensor from a global view. Must have size:
                (batch_size, global_view_height, global_view_width, global_feature_dim).
            z_b_local_features:
                A local features tensor from a global or local view. Must have size:
                (batch_size, height, width, global_feature_dim) where height and width
                depend on the view type.
            grid_a:
                A grid tensor from a global view. Must have size:
                (batch_size, grid_size, grid_size, 2).
            grid_b:
                A grid tensor from a global or local view. Must have size:
                (batch_size, grid_size, grid_size, 2).

        Returns:
            The loss.
        """

        if z_a_local_features.shape[0] < 1 or z_b_local_features.shape[0] < 1:
            raise ValueError(
                f"z_a_local and z_b_local must have batch size > 1 but found "
                f"{z_a_local_features.shape[0]} and {z_b_local_features.shape[0]}."
            )

        global_loss = self.vicregloss.forward(z_a=z_a, z_b=z_b)

        local_loss = self.local_loss(
            z_a=z_a_local_features,
            z_b=z_b_local_features,
            grid_a=grid_a,
            grid_b=grid_b,
        )

        loss = self.alpha * global_loss + (1 - self.alpha) * local_loss
        return loss
