from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

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
            Number of local features to match using nearest neighbors.

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
        alpha: float = 0.75,
        gather_distributed: bool = False,
        eps: float = 0.0001,
        num_matches: Tuple[int, int] = (20, 4),
    ):
        super(VICRegLLoss, self).__init__()
        self.alpha = alpha
        self.num_matches = num_matches
        self.vicregloss = VICRegLoss(
            lambda_param=lambda_param,
            mu_param=mu_param,
            nu_param=nu_param,
            eps=eps,
            gather_distributed=gather_distributed,
        )

    def _nearest_neighbors_on_l2(
        self, input_maps: Tensor, candidate_maps: Tensor, num_matches: int
    ) -> Tuple[Tensor, Tensor]:
        """
        input_maps: (B, H * W, C)
        candidate_maps: (B, H * W, C)

        Returns:
            (nn_input, nn_candidate) tuple containing two tensors with shape
            (B * num_matches, C).
        """
        distances = torch.cdist(input_maps, candidate_maps)
        nn_input, nn_candidate = nearest_neighbors(
            input_maps, candidate_maps, distances, num_matches
        )
        return nn_input.flatten(end_dim=1), nn_candidate.flatten(end_dim=1)

    def _nearest_neighbors_on_grid(
        self,
        input_grid: Tensor,
        candidate_grid: Tensor,
        input_maps: Tensor,
        candidate_maps: Tensor,
        num_matches: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        input_grid: (B, H * W, 2)
        candidate_grid: (B, H * W, 2)
        input_maps: (B, H * W, C)
        candidate_maps: (B, H * W, C)

        Returns:
            (nn_input, nn_candidate) tuple containing two tensors with shape
            (B * num_matches, C).
        """
        distances = torch.cdist(input_grid, candidate_grid)
        nn_input, nn_candidate = nearest_neighbors(
            input_maps, candidate_maps, distances, num_matches
        )
        return nn_input.flatten(end_dim=1), nn_candidate.flatten(end_dim=1)

    def local_loss(
        self,
        z_a: Tensor,
        z_b: Tensor,
        grid_a: Tensor,
        grid_b: Tensor,
        num_matches: int,
    ) -> Tensor:
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
            num_matches:
                Number of features to match using nearest neighbors.

        Returns:
            The local loss.
        """
        z_a = z_a.flatten(start_dim=1, end_dim=2)
        z_b = z_b.flatten(start_dim=1, end_dim=2)

        # L2 based loss
        z_a_filtered, z_a_nn = self._nearest_neighbors_on_l2(
            input_maps=z_a, candidate_maps=z_b, num_matches=num_matches
        )
        z_b_filtered, z_b_nn = self._nearest_neighbors_on_l2(
            input_maps=z_b, candidate_maps=z_a, num_matches=num_matches
        )
        l2_loss_a = self.vicregloss.forward(z_a=z_a_filtered, z_b=z_a_nn)
        l2_loss_b = self.vicregloss.forward(z_a=z_b_filtered, z_b=z_b_nn)
        l2_loss = (l2_loss_a + l2_loss_b) / 2

        # Grid based loss
        grid_a = grid_a.flatten(start_dim=1, end_dim=2)
        grid_b = grid_b.flatten(start_dim=1, end_dim=2)
        z_a_filtered, z_a_nn = self._nearest_neighbors_on_grid(
            input_grid=grid_a,
            candidate_grid=grid_b,
            input_maps=z_a,
            candidate_maps=z_b,
            num_matches=num_matches,
        )
        z_b_filtered, z_b_nn = self._nearest_neighbors_on_grid(
            input_grid=grid_b,
            candidate_grid=grid_a,
            input_maps=z_b,
            candidate_maps=z_a,
            num_matches=num_matches,
        )

        grid_loss_a = self.vicregloss.forward(z_a=z_a_filtered, z_b=z_a_nn)
        grid_loss_b = self.vicregloss.forward(z_a=z_b_filtered, z_b=z_b_nn)
        grid_loss = (grid_loss_a + grid_loss_b) / 2
        return l2_loss + grid_loss

    def forward(
        self,
        global_view_features: Sequence[Tuple[Tensor, Tensor]],
        global_view_grids: Sequence[Tensor],
        local_view_features: Optional[Sequence[Tuple[Tensor, Tensor]]] = None,
        local_view_grids: Optional[Sequence[Tensor]] = None,
    ) -> Tensor:
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
        if len(global_view_features) != len(global_view_grids):
            raise ValueError(
                f"global_view_features and global_view_grids must have same length "
                f"but found {len(global_view_features)} and {len(global_view_grids)}."
            )
        if local_view_features is not None and local_view_grids is not None:
            if len(local_view_features) != len(local_view_grids):
                raise ValueError(
                    f"local_view_features and local_view_grids must have same length "
                    f"but found {len(local_view_features)} and {len(local_view_grids)}."
                )
        elif local_view_features is not None or local_view_grids is not None:
            raise ValueError(
                f"local_view_features and local_view_grids must either both be set or "
                f"None but found {type(local_view_features)} and {type(local_view_grids)}."
            )

        # calculate global features loss
        global_features_loss = 0
        global_loss_count = 0
        for z_a_global_features, _ in global_view_features:
            # global views
            for z_b_global_features, _ in global_view_features:
                if z_a_global_features is not z_b_global_features:
                    global_features_loss += self.vicregloss.forward(
                        z_a=z_a_global_features, z_b=z_b_global_features
                    )
                    global_loss_count += 1

            # local views
            if local_view_features is not None:
                for z_b_global_features, _ in local_view_features:
                    global_features_loss += self.vicregloss.forward(
                        z_a=z_a_global_features, z_b=z_b_global_features
                    )
                    global_loss_count += 1
        global_features_loss /= global_loss_count

        # calculate local features loss
        local_features_loss = 0
        local_loss_count = 0
        for (_, z_a_local_features), grid_a in zip(
            global_view_features, global_view_grids
        ):
            # global views
            for (_, z_b_local_features), grid_b in zip(
                global_view_features, global_view_grids
            ):
                if z_a_local_features is not z_b_local_features:
                    local_features_loss += self.local_loss(
                        z_a=z_a_local_features,
                        z_b=z_b_local_features,
                        grid_a=grid_a,
                        grid_b=grid_b,
                        num_matches=self.num_matches[0],
                    )
                    local_loss_count += 1
            # local views
            if local_view_features is not None and local_view_grids is not None:
                for (_, z_b_local_features), grid_b in zip(
                    local_view_features, local_view_grids
                ):
                    local_features_loss += self.local_loss(
                        z_a=z_a_local_features,
                        z_b=z_b_local_features,
                        grid_a=grid_a,
                        grid_b=grid_b,
                        num_matches=self.num_matches[1],
                    )
                    local_loss_count += 1
        local_features_loss /= local_loss_count

        loss = (
            self.alpha * global_features_loss + (1 - self.alpha) * local_features_loss
        )
        return loss
