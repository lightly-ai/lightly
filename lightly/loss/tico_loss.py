import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightly.utils.dist import gather

class TiCoLoss(torch.nn.Module):
    """Implementation of the Tico Loss from Tico[0] paper.
    This implementation takes inspiration from the code published 
    by sayannag using Lightly. [1]

    [0] Jiachen Zhu et. al, 2022, Tico... https://arxiv.org/abs/2206.10698
    [1] https://github.com/sayannag/TiCo-pytorch
        
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
        beta_param: float = 0.9,
        ro_param: float = 20.0,
        gather_distributed : bool = False,
    ):
        """Lambda, mu and nu params configuration with default value like in [0]
        Args:
            beta_param:
                Coefficient for the EMA update of the covariance
                Defaults to 0.9 [0].
            ro_param:
                Weight for the covariance term of the loss
                Defaults to 20.0 [0].
            gather_distributed:
                If True then the cross-correlation matrices from all gpus are
                gathered and summed before the loss calculation.
        """
        super(TiCoLoss, self).__init__()
        self.beta_param = beta_param
        self.ro_param = ro_param
        self.gather_distributed = gather_distributed

    def forward(self, C: torch.Tensor, z_a: torch.Tensor, z_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert z_a.shape[0] > 1 and z_b.shape[0] > 1, f"z_a and z_b must have batch size > 1 but found {z_a.shape[0]} and {z_b.shape[0]}"
        assert z_a.shape == z_b.shape, f"z_a and z_b must have same shape but found {z_a.shape} and {z_b.shape}."
        

        # gather all batches
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                z_a = torch.cat(gather(z_a), dim=0)
                z_b = torch.cat(gather(z_b), dim=0)

        # normalize repr. along the batch dimension
        z_a = z_a - z_a.mean(0) # NxD
        z_b = z_b - z_b.mean(0) # NxD

        # normalize image
        z_a = torch.nn.functional.normalize(z_a, dim = -1)
        z_b = torch.nn.functional.normalize(z_b, dim = -1)
        
        # compute loss
        B = torch.mm(z_a.T, z_a)/z_a.shape[0]        
        C = self.beta_param * C + (1 - self.beta_param) * B
        loss = 1 - (z_a * z_b).sum(dim=1).mean() + self.ro_param * (torch.mm(z_a, C) * z_a).sum(dim=1).mean()

        return loss, C
