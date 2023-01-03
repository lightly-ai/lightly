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
        num_matches: List[int] = [20, 4]
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
        self,
        input_maps: torch.Tensor, 
        candidate_maps: torch.Tensor, 
        num_matches: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_maps: (B, C * H, W)
        candidate_maps: (B, C * H, W)
        """
        distances = torch.cdist(input_maps, candidate_maps)


        return nearest_neighbors(input_maps, candidate_maps, distances, num_matches)
    
    def _nearest_neighbors_on_location(
        self,
        input_location: torch.Tensor,
        candidate_location: torch.Tensor, 
        input_maps: torch.Tensor, 
        candidate_maps: torch.Tensor, 
        num_matches: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        input_location: (B, H * W, 2)
        candidate_location: (B, H * W, 2)
        input_maps: (B, H * W, C)
        candidate_maps: (B, H * W, C)
        """
    
        
        distances = torch.cdist(input_location, candidate_location)
        
        return nearest_neighbors(input_maps, candidate_maps, distances, num_matches)


    def localLoss(self, 
        z_a: torch.Tensor, 
        z_b: torch.Tensor, 
        location_a: torch.Tensor, 
        location_b: torch.Tensor
    ) -> torch.Tensor:

        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0

        z_a = z_a.flatten(1, 2)
        z_b = z_b.flatten(1, 2)
        # print(z_a.shape)
        # L2 loss
        
        z_0_filtered, z_0_nn = self._nearest_neighbors_on_l2(
            input_maps=z_a, candidate_maps=z_b, num_matches=self.num_matches[0]
        )
        
        z_1_filtered, z_1_nn = self._nearest_neighbors_on_l2(
            input_maps=z_b, candidate_maps=z_a, num_matches=self.num_matches[1]
        )
        

        # add fast version from paper implementation

        inv_loss_1 = F.mse_loss(z_0_filtered, z_0_nn)
        inv_loss_2 = F.mse_loss(z_1_filtered, z_1_nn)
        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

        location_a = location_a.flatten(1, 2)
        location_b = location_b.flatten(1, 2)
        location_c = copy.deepcopy(location_a)
        location_d = copy.deepcopy(location_b)


        # distance loss

        z_2_filtered, z_2_nn = self._nearest_neighbors_on_location(
            input_location=location_a,
            candidate_location=location_b,
            input_maps=z_a,
            candidate_maps=z_b,
            num_matches=self.num_matches[0]
        )

        z_3_filtered, z_3_nn = self._nearest_neighbors_on_location(
            input_location=location_d,
            candidate_location=location_c,
            input_maps=z_b,
            candidate_maps=z_a,
            num_matches=self.num_matches[1]
        )

        inv_loss_1 = F.mse_loss(z_2_filtered, z_2_nn)
        inv_loss_2 = F.mse_loss(z_3_filtered, z_3_nn)
        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

        local_loss = self.lambda_param * inv_loss + self.mu_param * var_loss + self.nu_param * cov_loss
        return local_loss

    def forward(self,
        z_a: torch.Tensor, 
        z_b: torch.Tensor, 
        z_a_local: torch.Tensor, 
        z_b_local: torch.Tensor, 
        location_a: torch.Tensor, 
        location_b: torch.Tensor
    ) -> torch.Tensor:

        assert z_a_local.shape[0] > 1 and z_b_local.shape[0] > 1, f"z_a_local and z_b_local must have batch size > 1 but found {z_a_local.shape[0]} and  {z_b_local.shape[0]}"

        global_loss = self.vicregloss.forward(z_a=z_a, z_b=z_b)

        local_loss = self.localLoss(z_a=z_a_local, z_b=z_b_local, location_a=location_a, location_b=location_b)

        loss = self.alpha * global_loss + (1 - self.alpha) * local_loss

        return loss
