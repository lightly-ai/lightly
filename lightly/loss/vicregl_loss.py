from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module

from lightly.loss.vicreg_loss import (
    VICRegLoss,
    covariance_loss,
    invariance_loss,
    variance_loss,
)
from lightly.models import utils
from lightly.utils.dist import gather


class VICRegLLoss(Module):
    """Implementation of the VICRegL loss from VICRegL paper [0].

    This implementation follows the code published by the authors [1].

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
            If True, the cross-correlation matrices from all gpus are gathered and
            summed before the loss calculation.
        eps:
            Epsilon for numerical stability.
        num_matches:
            Number of local features to match using nearest neighbors.

    Examples:
        >>> # initialize loss function
        >>> criterion = VICRegLLoss()
        >>> transform = VICRegLTransform(n_global_views=2, n_local_views=4)
        >>>
        >>> # generate two random transforms of images
        >>> views_and_grids = transform(images)
        >>> views = views_and_grids[:6] # 2 global views + 4 local views
        >>> grids = views_and_grids[6:]
        >>>
        >>> # feed through model images
        >>> features = [model(view) for view in views]
        >>>
        >>> # calculate loss
        >>> loss = criterion(
        ...     global_view_features=features[:2],
        ...     global_view_grids=grids[:2],
        ...     local_view_features=features[2:],
        ...     local_view_grids=grids[2:],
        ... )
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
        """Initializes the VICRegL loss module with the specified parameters.

        Raises:
            ValueError: If gather_distributed is True but torch.distributed is not available.
        """
        super(VICRegLLoss, self).__init__()
        self.alpha = alpha
        self.num_matches = num_matches
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.eps = eps
        self.gather_distributed = gather_distributed
        # Note: We multiply nu_param by 0.5 because the implementations of the VICReg
        # covariance loss differ by a factor of 0.5 between the original VICReg and
        # VICRegL codebases. See:
        # - VICReg: https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/main_vicreg.py#L211-L213
        # - VICRegL: https://github.com/facebookresearch/VICRegL/blob/803ae4c8cd1649a820f03afb4793763e95317620/main_vicregl.py#L308-L312
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
        """Computes the global and local VICRegL loss from the input features.

        Args:
            global_view_features:
                Sequence of (global_features, local_features) tuples from the global
                crop views. global_features must have size
                (batch_size, global_feature_dim) and local_features must have size
                (batch_size, grid_height, grid_width, local_feature_dim).
            global_view_grids:
                Sequence of grid tensors from the global crop views. Every tensor must
                have shape (batch_size, grid_height, grid_width, 2).
            local_view_features:
                Sequence of (global_features, local_features) tuples from the local crop
                views. global_features must have size
                (batch_size, global_feature_dim) and local_features must have size
                (batch_size, grid_height, grid_width, local_feature_dim). Note that
                grid_height and grid_width can differ between global_view_features and
                local_view_features.
            local_view_grids:
                Sequence of grid tensors from the local crop views. Every tensor must
                have shape (batch_size, grid_height, grid_width, 2). Note that
                grid_height and grid_width can differ between global_view_features and
                local_view_features.

        Returns:
            Weighted sum of the global and local loss, calculated as:
            `self.alpha * global_loss + (1-self.alpha) * local_loss`.

        Raises:
            ValueError: If the lengths of global_view_features and global_view_grids are not the same.
            ValueError: If the lengths of local_view_features and local_view_grids are not the same.
            ValueError: If only one of local_view_features or local_view_grids is set.
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

        # Calculate loss from global features
        global_loss = self._global_loss(
            global_view_features=global_view_features,
            local_view_features=local_view_features,
        )

        # Calculate loss from local features
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
        """Returns global features loss.

        Args:
        global_view_features:
                Sequence of (global_features, local_features)
                tuples from the global crop views.
        local_view_features:
                Sequence of (global_features,local_features)
                tuples from the local crop views.

        Returns:
            The computed global features loss.
        """
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
        """Returns invariance loss from global features.

        Args:
            global_view_features:
                        Sequence of (global_features, local_features)
                        tuples from the global crop views.
            local_view_features:
                        Sequence of (global_features,local_features)
                        tuples from the local crop views.

        Returns:
            The computed invariance loss from global features.
        """
        loss = torch.tensor(0.0)
        loss_count = torch.tensor(0)

        # Compute invariance loss between global views
        for global_features_a, _ in global_view_features:
            for global_features_b, _ in global_view_features:
                if global_features_a is not global_features_b:
                    loss += invariance_loss(global_features_a, global_features_b)
                    loss_count += 1

            # Compute invariance loss between global and local views
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
        """Returns variance and covariance loss from global features.

        Args:
            global_view_features: Sequence of (global_features, local_features)
                    tuples from the global crop views.
            local_view_features: Sequence of (global_features,local_features)
                    tuples from the local crop views.

        Returns:
            The computed variance and covariance loss from global features.
        """
        view_features = list(global_view_features)
        if local_view_features is not None:
            view_features = view_features + list(local_view_features)

        var_loss = torch.tensor(0.0)
        cov_loss = torch.tensor(0.0)
        loss_count = torch.tensor(0)
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
        """Returns loss from local features based on nearest neighbor matching.

        Note: Our nearest neighbor implementation returns the selected features sorted
        by increasing matching distance, whereas the implementation by the VICRegL
        authors returns features in a different order [1]. This results in slight
        differences of the final local loss. The difference results from feature
        centering which depends on the order.

        Note: Nearest neighbor matching slightly differs between the paper [0] and the
        original implementation of the authors [1]. The paper mentions that
        num_matches is set to 20 for global views and 4 for local views. The code
        uses 20 matches for the first NN search and 4 matches for the second search,
        regardless of global or local views:
        https://github.com/facebookresearch/VICRegL/blob/803ae4c8cd1649a820f03afb4793763e95317620/main_vicregl.py#L329-L334
        Our implementation follows the original code and ignores view type.

        Args:
            global_view_features:
                Sequence of (global_features, local_features) tuples from the global crop views.
            global_view_grids:
                Sequence of grid tensors from the global crop views.
            local_view_features:
                Sequence of (global_features,local_features) tuples from the local crop views.
            local_view_grids:
                Sequence of grid tensors from the local crop views.

        Returns:
            The computed loss from local features based on nearest neighbor matching.
        """
        loss = torch.tensor(0.0)
        loss_count = torch.tensor(0)

        # Compute the loss for global views
        for (_, z_a_local_features), grid_a in zip(
            global_view_features, global_view_grids
        ):
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

            # Compute the loss for local views
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
        """Returns loss for local features matched with neareast neighbors using L2
        distance in the feature space.

        Args:
            z_a:
                Local feature tensor with shape (batch_size, height, width, dim).
            z_b:
                Local feature tensor with shape (batch_size, height, width, dim).

        Returns:
            The computed loss for local features.
        """
        # (batch_size, height, width, dim) -> (batch_size, height * width, dim)
        z_a = z_a.flatten(start_dim=1, end_dim=2)
        z_b = z_b.flatten(start_dim=1, end_dim=2)

        # Find nearest neighbours using L2 distance
        z_a_filtered, z_a_nn = self._nearest_neighbors_on_l2(
            input_features=z_a, candidate_features=z_b, num_matches=self.num_matches[0]
        )
        z_b_filtered, z_b_nn = self._nearest_neighbors_on_l2(
            input_features=z_b, candidate_features=z_a, num_matches=self.num_matches[1]
        )

        # Compute VICReg losses
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
        """Returns loss for local features matched with nearest neighbors based on
        the feature location.

        Args:
            z_a:
                Local feature tensor with shape (batch_size, height, width, dim).
            z_b:
                Local feature tensor with shape (batch_size, height, width, dim).
                Note that height and width can be different from z_a.
            grid_a:
                Grid tensor with shape (batch_size, height, width, 2).
            grid_b:
                Grid tensor with shape (batch_size, height, width, 2).
                Note that height and width can be different from grid_a.

        Returns:
            The computed loss for local features based on nearest neighbour matching.
        """
        # (batch_size, height, width, dim) -> (batch_size, height * width, dim)
        z_a = z_a.flatten(start_dim=1, end_dim=2)
        z_b = z_b.flatten(start_dim=1, end_dim=2)

        # (batch_size, height, width, 2) -> (batch_size, height * width, 2)
        grid_a = grid_a.flatten(start_dim=1, end_dim=2)
        grid_b = grid_b.flatten(start_dim=1, end_dim=2)

        # Find nearest neighbours based on grid location
        z_a_filtered, z_a_nn = self._nearest_neighbors_on_grid(
            input_features=z_a,
            candidate_features=z_b,
            input_grid=grid_a,
            candidate_grid=grid_b,
            num_matches=self.num_matches[0],
        )
        z_b_filtered, z_b_nn = self._nearest_neighbors_on_grid(
            input_features=z_b,
            candidate_features=z_a,
            input_grid=grid_b,
            candidate_grid=grid_a,
            num_matches=self.num_matches[1],
        )

        # Compute VICReg losses
        loss_a = self.vicreg_loss.forward(z_a=z_a_filtered, z_b=z_a_nn)
        loss_b = self.vicreg_loss.forward(z_a=z_b_filtered, z_b=z_b_nn)
        return 0.5 * (loss_a + loss_b)

    def _nearest_neighbors_on_l2(
        self, input_features: Tensor, candidate_features: Tensor, num_matches: int
    ) -> Tuple[Tensor, Tensor]:
        """Finds num_matches closest neighbors of input_features in candidate_features.

        Args:
            input_features:
                Local features tensor with shape (batch_size, height * width, dim).
            candidate_features:
                Local features tensor with shape (batch_size, height * width, dim).
                Note that height and width can be different from input_features.

        Returns:
            (nn_input, nn_candidate) tuple containing two tensors with shape
            (batch_size, num_matches, dim).
        """
        distances = torch.cdist(input_features, candidate_features)
        # TODO(Philipp, 12/24): Remove type ignore when utils are typechecked.
        return utils.nearest_neighbors(  # type: ignore[no-any-return]
            input_features, candidate_features, distances, num_matches
        )

    def _nearest_neighbors_on_grid(
        self,
        input_features: Tensor,
        candidate_features: Tensor,
        input_grid: Tensor,
        candidate_grid: Tensor,
        num_matches: int,
    ) -> Tuple[Tensor, Tensor]:
        """Finds num_matches closest neighbors of input_features in candidate_features
        based on the distance between the features defined by input_grid and
        candidate_grid.

        Args:
            input_features:
                Local features tensor with shape (batch_size, height * width, dim).
            candidate_features:
                Local features tensor with shape (batch_size, height * width, dim).
                Note that height and width can be different from input_features.
            input_grid:
                Grid tensor with shape (batch_size, height, width, 2).
            candidate_grid:
                Grid tensor with shape (batch_size, height, width, 2). Note that height
                and width can be different from input_grid.

        Returns:
            (nn_input, nn_candidate) tuple containing two tensors with shape
            (batch_size, num_matches, dim).
        """
        distances: Tensor = torch.cdist(input_grid, candidate_grid)
        # TODO(Philipp, 12/24): Remove type ignore when utils are typechecked.
        return utils.nearest_neighbors(  # type: ignore[no-any-return]
            input_features, candidate_features, distances, num_matches
        )
