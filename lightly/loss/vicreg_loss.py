import torch
import torch.distributed as dist
import torch.nn.functional as F

class VICRegLoss(torch.nn.Module):
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

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:

        # invariance term of the loss
        repr_loss = F.mse_loss(z_a, z_b)

        # gather all batches
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                dist.all_reduce(z_a)
                dist.all_reduce(z_b)

        # normalize repr. along the batch dimension
        z_a = z_a - z_a.mean(0) # NxD
        z_b = z_b - z_b.mean(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # variance term of the loss
        std_x = torch.sqrt(z_a.var(dim=0) + self.eps)
        std_y = torch.sqrt(z_b.var(dim=0) + self.eps)
        std_loss = 0.5 * (torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y)))

        # covariance term of the loss
        cov_x = (z_a.T @ z_a) / (N - 1)
        cov_y = (z_b.T @ z_b) / (N - 1)

        # compute off-diagonal elements
        n, _ = cov_x.shape
        off_diag_cov_x = cov_x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        off_diag_cov_y = cov_y.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        cov_loss = off_diag_cov_x.pow_(2).sum().div(D) + off_diag_cov_y.pow_(2).sum().div(D)

        # loss
        loss = self.lambda_param * repr_loss + self.mu_param * std_loss + self.nu_param * cov_loss

        return loss
