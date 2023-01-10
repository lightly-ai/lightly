import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightly.utils.dist import gather
from lightly.loss.vicreg_loss import VICRegLoss
from lightly.models.utils import nearest_neighbors
from typing import List, Tuple
import copy


class VICRegLLoss(torch.nn.Module):
    """Implementation of the VICRegL Loss from VICRegL[0] paper.
    This implementation follows the code published by the authors. [1]

    [0] Bardes, A. et. al, 2022, VICReg... https://arxiv.org/abs/2210.01571
    [1] https://github.com/facebookresearch/VICRegL

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
        num_matches: Tuple[int] = (20, 4),
    ):
        """Lambda, mu and nu params configuration with default value like in [0]
        Args:
            lambda_param:
                Coefficient for the invariance term of the loss
                Defaults to 25.0 [0].
            mu_param:
                Coefficient for the variance term of the loss
                Defaults to 25.0 [0].
            nu_param:
                Coefficient for the covariance term of the loss
                Defaults to 1.0 [0].
            alpha:
                Coefficient to weight local loss with global loss
            gather_distributed:
                If True then the cross-correlation matrices from all gpus are
                gathered and summed before the loss calculation.
            eps:
                Numerical epsilon
                Defaults to 0.0001 [1].
            num_matches:
                Number of local matches between patches in the KNN
        """
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
        z_global: torch.Tensor,
        z_local: torch.Tensor,
        grid_global: torch.Tensor,
        grid_local: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the local loss

        Compute the local loss between two sets of maps using nearest neighbors and location loss.

        Args:
            z_global:
                A tensor of global maps. It has size: [batch_size, global_image_height_crop, global_image_width_crop, global_feature_dimension]
            z_local:
                A tensor of local maps. It has size: [batch_size, local_image_height_crop, local_image_width_crop, local_feature_dimension]
            grid_global:
                A tensor of grids for the global maps. It has size: [batch_size, grid_size, grid_size, 2]
            grid_local:
                A tensor of grids for the local maps. It has size: [batch_size, grid_size, grid_size, 2]

        Returns:
            A tensor of the local loss between the two sets of maps. It has size: [batch_size]"""

        inv_loss = 0.0

        z_global = z_global.flatten(1, 2)
        z_local = z_local.flatten(1, 2)

        # L2 loss

        z_global_filtered, z_global_nn = self._nearest_neighbors_on_l2(
            input_maps=z_global, candidate_maps=z_local, num_matches=self.num_matches[0]
        )

        z_local_filtered, z_local_nn = self._nearest_neighbors_on_l2(
            input_maps=z_local, candidate_maps=z_global, num_matches=self.num_matches[1]
        )

        # add fast version from paper implementation

        inv_loss_global = F.mse_loss(z_global_filtered, z_global_nn)
        inv_loss_local = F.mse_loss(z_local_filtered, z_local_nn)
        inv_loss = inv_loss + (inv_loss_global / 2 + inv_loss_local / 2)

        grid_global = grid_global.flatten(1, 2)
        grid_local = grid_local.flatten(1, 2)

        # distance loss

        z_global_filtered, z_global_nn = self._nearest_neighbors_on_grid(
            input_grid=grid_global,
            candidate_grid=grid_local,
            input_maps=z_global,
            candidate_maps=z_local,
            num_matches=self.num_matches[0],
        )

        z_local_filtered, z_local_nn = self._nearest_neighbors_on_grid(
            input_grid=grid_local,
            candidate_grid=grid_global,
            input_maps=z_local,
            candidate_maps=z_global,
            num_matches=self.num_matches[1],
        )

        inv_loss_global = F.mse_loss(z_global_filtered, z_global_nn)
        inv_loss_local = F.mse_loss(z_local_filtered, z_local_nn)
        inv_loss = inv_loss + (inv_loss_global / 2 + inv_loss_local / 2)

        local_loss = self.lambda_param * inv_loss

        return local_loss

    def forward(
        self,
        z_global: torch.Tensor,
        z_local: torch.Tensor,
        z_global_local_features: torch.Tensor,
        z_local_local_features: torch.Tensor,
        grid_global: torch.Tensor,
        grid_local: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the overall loss between two sets of maps, using global loss and local loss.

        It computes global loss using the VICReg loss module and z_global and z_local, and local loss .
        It then combines the global and local loss using a scalar value alpha, and returns the result as loss.

        Args:
            z_global:
                A tensor of global maps. It has size: [batch_size, global_image_height, global_image_width, global_feature_dimension]
            z_local:
                A tensor of local maps. It has size: [batch_size, local_image_height, local_image_width, local_feature_dimension]
            z_global_local_features:
                A tensor of local features for the global maps. It has size: [batch_size, global_image_height_crop, global_image_width_crop, global_feature_dimension]
            z_local_local_features:
                A tensor of local features for the local maps. It has size: [batch_size, global_image_height_crop, global_image_width_crop, global_feature_dimension]
            grid_global:
                A tensor of grids for the global maps. It has size: [batch_size, grid_size, grid_size, 2]
            grid_local:
                A tensor of grids for the local maps. It has size: [batch_size, grid_size, grid_size, 2]

        Returns:
            A tensor of the overall loss between the two sets of maps. It has size: [batch_size]"
        """

        if z_global_local_features.shape[0] < 1 or z_local_local_features.shape[0] < 1:
            raise ValueError(
                f"z_global_local and z_local_local must have batch size > 1 but found {z_global_local_features.shape[0]} and  {z_local_local_features.shape[0]}"
            )

        global_loss = self.vicregloss.forward(z_a=z_global, z_b=z_local)

        local_loss = self.local_loss(
            z_global=z_global_local_features,
            z_local=z_local_local_features,
            grid_global=grid_global,
            grid_local=grid_local,
        )

        loss = self.alpha * global_loss + (1 - self.alpha) * local_loss

        return loss
