from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, dist

from lightly.loss.vicreg_loss import (
    VICRegLoss,
    covariance_loss,
    invariance_loss,
    variance_loss,
)
from lightly.models.utils import nearest_neighbors
from lightly.utils.dist import gather


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
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.eps = eps
        self.gather_distributed = gather_distributed
        self.vicreg_loss = VICRegLoss(
            lambda_param=lambda_param,
            mu_param=mu_param,
            nu_param=0.5 * nu_param,
            eps=eps,
            gather_distributed=gather_distributed,
        )

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

        # calculate loss from global features
        global_loss = self._global_loss(
            global_view_features=global_view_features,
            local_view_features=local_view_features,
        )

        # calculate loss from local features
        local_loss = self._local_loss(
            global_view_features=global_view_features,
            global_view_grids=global_view_grids,
            local_view_features=local_view_features,
            local_view_grids=local_view_grids,
        )

        loss = self.alpha * global_loss + (1 - self.alpha) * local_loss
        return loss

    def _global_loss(
        self,
        global_view_features: Sequence[Tuple[Tensor, Tensor]],
        local_view_features: Optional[Sequence[Tuple[Tensor, Tensor]]] = None,
    ) -> Tensor:
        inv_loss = self._global_invariance_loss(
            global_view_features=global_view_features,
            local_view_features=local_view_features,
        )
        var_loss, cov_loss = self._global_variance_and_covariance_loss(
            global_view_features=global_view_features,
            local_view_features=local_view_features,
        )
        return (
            self.lambda_param * inv_loss
            + self.mu_param * var_loss
            + self.nu_param * cov_loss
        )

    def _global_invariance_loss(
        self,
        global_view_features: Sequence[Tuple[Tensor, Tensor]],
        local_view_features: Optional[Sequence[Tuple[Tensor, Tensor]]] = None,
    ) -> Tensor:
        loss = 0
        loss_count = 0
        for global_features_a, _ in global_view_features:
            # global views
            for global_features_b, _ in global_view_features:
                if global_features_a is not global_features_b:
                    loss += invariance_loss(global_features_a, global_features_b)
                    loss_count += 1

            # local views
            if local_view_features is not None:
                for global_features_b, _ in local_view_features:
                    loss += invariance_loss(global_features_a, global_features_b)
                    loss_count += 1
        return loss / loss_count

    def _global_variance_and_covariance_loss(
        self,
        global_view_features: Sequence[Tuple[Tensor, Tensor]],
        local_view_features: Optional[Sequence[Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor]:
        view_features = list(global_view_features)
        if local_view_features is not None:
            view_features = view_features + list(local_view_features)

        var_loss = 0
        cov_loss = 0
        loss_count = 0
        for global_features, _ in view_features:
            if self.gather_distributed and dist.is_initialized():
                world_size = dist.get_world_size()
                if world_size > 1:
                    global_features = torch.cat(gather(global_features), dim=0)

            var_loss += variance_loss(x=global_features, eps=self.eps)
            cov_loss += covariance_loss(x=global_features)
            loss_count += 1
        return var_loss / loss_count, cov_loss / loss_count

    def _local_loss(
        self,
        global_view_features: Sequence[Tuple[Tensor, Tensor]],
        global_view_grids: Sequence[Tensor],
        local_view_features: Optional[Sequence[Tuple[Tensor, Tensor]]] = None,
        local_view_grids: Optional[Sequence[Tensor]] = None,
    ) -> Tensor:
        loss = 0
        loss_count = 0
        for (_, z_a_local_features), grid_a in zip(
            global_view_features, global_view_grids
        ):
            # global views
            for (_, z_b_local_features), grid_b in zip(
                global_view_features, global_view_grids
            ):
                if z_a_local_features is not z_b_local_features:
                    loss += self._local_l2_loss(
                        z_a=z_a_local_features,
                        z_b=z_b_local_features,
                    )
                    loss += self._local_location_loss(
                        z_a=z_a_local_features,
                        z_b=z_b_local_features,
                        grid_a=grid_a,
                        grid_b=grid_b,
                    )
                    loss_count += 1

            # local views
            if local_view_features is not None and local_view_grids is not None:
                for (_, z_b_local_features), grid_b in zip(
                    local_view_features, local_view_grids
                ):
                    loss += self._local_l2_loss(
                        z_a=z_a_local_features,
                        z_b=z_b_local_features,
                    )
                    loss += self._local_location_loss(
                        z_a=z_a_local_features,
                        z_b=z_b_local_features,
                        grid_a=grid_a,
                        grid_b=grid_b,
                    )
                    loss_count += 1
        return loss / loss_count

    def _local_l2_loss(
        self,
        z_a: Tensor,
        z_b: Tensor,
    ) -> Tensor:
        z_a = z_a.flatten(start_dim=1, end_dim=2)
        z_b = z_b.flatten(start_dim=1, end_dim=2)

        z_a_filtered, z_a_nn = self._nearest_neighbors_on_l2(
            input_maps=z_a, candidate_maps=z_b, num_matches=self.num_matches[0]
        )
        z_b_filtered, z_b_nn = self._nearest_neighbors_on_l2(
            input_maps=z_b, candidate_maps=z_a, num_matches=self.num_matches[1]
        )
        loss_a = self.vicreg_loss.forward(z_a=z_a_filtered, z_b=z_a_nn)
        loss_b = self.vicreg_loss.forward(z_a=z_b_filtered, z_b=z_b_nn)
        return 0.5 * (loss_a + loss_b)

    def _local_location_loss(
        self,
        z_a: Tensor,
        z_b: Tensor,
        grid_a: Tensor,
        grid_b: Tensor,
    ) -> Tensor:
        z_a = z_a.flatten(start_dim=1, end_dim=2)
        z_b = z_b.flatten(start_dim=1, end_dim=2)
        grid_a = grid_a.flatten(start_dim=1, end_dim=2)
        grid_b = grid_b.flatten(start_dim=1, end_dim=2)
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

        loss_a = self.vicreg_loss.forward(z_a=z_a_filtered, z_b=z_a_nn)
        loss_b = self.vicreg_loss.forward(z_a=z_b_filtered, z_b=z_b_nn)
        return 0.5 * (loss_a + loss_b)

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
        return nearest_neighbors(input_maps, candidate_maps, distances, num_matches)

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
        return nearest_neighbors(input_maps, candidate_maps, distances, num_matches)
