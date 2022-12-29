import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightly.utils.dist import gather
from lightly.loss.vicreg_loss import VICRegLoss



class VICRegLLoss(torch.nn.Module):
    """Implementation of the VICReg Loss from VICReg[0] paper.
    This implementation follows the code published by the authors. [1]

    [0] Bardes, A. et. al, 2022, VICReg... https://arxiv.org/abs/2105.04906
    [1] https://github.com/facebookresearch/vicreg/
        
    Examples:
    
        >>> # initialize loss function
        >>> loss_fn = VICRegLoss()
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
        lambda_param: float = 25.0,
        mu_param: float = 25.0,
        nu_param: float = 1.0,
        gather_distributed : bool = False,
        eps = 0.0001
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
            gather_distributed:
                If True then the cross-correlation matrices from all gpus are
                gathered and summed before the loss calculation.
            eps:
                Numerical epsilon
                Defaults to 0.0001 [1].
        """
        super(VICRegLoss, self).__init__()
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        self.gather_distributed = gather_distributed

        self.eps = eps

    

    def localLoss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:

        return

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, z_a_local: torch.Tensor, z_b_local: torch.Tensor) -> torch.Tensor:
        assert z_a_local.shape[0] > 1 and z_b_local.shape[0] > 1, f"z_a_local and z_b_local must have batch size > 1 but found {z_a_local.shape[0]} and  {z_b_local.shape[0]}"
        global_criterion = VICRegLoss()
        global_loss = global_criterion(z_a, z_b)

        local_loss = self.localLoss(z_a_local, z_b_local)

        loss = global_loss + local_loss
        return loss
