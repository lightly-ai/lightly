"""
FIXME: hypersphere is perhaps bad naming as I am not sure it is the essence;
 alignment-and-uniformity loss perhaps? Does not sound as nice.
"""

import torch
import torch.nn.functional as F


class HypersphereLoss(torch.nn.Module):
    """
    Implementation of the loss described in 'Understanding Contrastive Representation Learning through
    Alignment and Uniformity on the Hypersphere.' [0]
    
    [0] Tongzhou Wang. et.al, 2020, ... https://arxiv.org/abs/2005.10242

    Note:
        In order for this loss to function as advertized, an l1-normalization to the hypersphere is required.
        This loss function applies this l1-normalization internally in the loss-layer.
        However, it is recommended that the same normalization is also applied in your architecture,
        considering that this l1-loss is also intended to be applied during inference.
        Perhaps there may be merit in leaving it out of the inferrence pathway, but this use has not been tested.

        Moreover it is recommended that the layers preceeding this loss function are either a linear layer without activation,
        a batch-normalization layer, or both. The directly upstream architecture can have a large influence
        on the ability of this loss to achieve its stated aim of promoting uniformity on the hypersphere;
        and if by contrast the last layer going into the embedding is a RELU or similar nonlinearity,
        we may see that we will never get very close to achieving the goal of uniformity on the hypersphere,
        but will confine ourselves to the subspace of positive activations.
        Similar architectural considerations are relevant to most contrastive loss functions,
        but we call it out here explicitly.

    Examples:
        >>> # initialize loss function
        >>> loss_fn = HypersphereLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimSiam model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)

    """

    def __init__(self, t=1., lam=1., alpha=2.):
        super(HypersphereLoss, self).__init__()
        self.t = t
        self.lam = lam
        self.alpha = alpha

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor, [b, d], float)
            y (torch.Tensor, [b, d], float)

        Returns:
            Loss (torch.Tensor, [], float)

        """
        x = F.normalize(z_a)
        y = F.normalize(z_b)

        def lalign(x, y):
            return (x - y).norm(dim=1).pow(self.alpha).mean()
        def lunif(x):
            sq_pdist = torch.pdist(x, p=2).pow(2)
            return sq_pdist.mul(-self.t).exp().mean().log()
        return lalign(x, y) + self.lam * (lunif(x) + lunif(y)) / 2
