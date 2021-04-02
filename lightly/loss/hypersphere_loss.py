"""
https://arxiv.org/pdf/2005.10242.pdf

NOTE: hypersphere is perhaps bad naming as I am not sure it is the essence;
alignment-and-uniformity loss perhaps? Does not sound as nice.
"""

import torch


def normalize_l2(x, dim=-1):
    return x / x.norm(p=2, dim=dim, keepdim=True)


class HypersphereLoss(torch.nn.Module):
    """

    Implementation of the loss described in 'Understanding Contrastive Representation Learning through
    Alignment and Uniformity on the Hypersphere.' [0]
    [0] Tongzhou Wang. et.al, 2020, ... https://arxiv.org/pdf/2005.10242.pdf

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
        """Parameters as described in [0]
        """
        super(HypersphereLoss, self).__init__()
        self.t = t
        self.lam = lam
        self.alpha = alpha

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x: torch.Tensor, [b, d], float
        y: torch.Tensor,[b, d], float

        Returns
        -------
        torch.Tensor, [], float
            loss

        """
        # FIXME: this is necessary for the loss to function as advertized,
        #  but not technically part of the loss i feel
        #  as it needs to be applied during inference as well
        #  leave that responsibility to the end user?
        x = normalize_l2(z_a)
        y = normalize_l2(z_b)

        def lalign(x, y):
            return (x - y).norm(dim=1).pow(self.alpha).mean()
        def lunif(x):
            sq_pdist = torch.pdist(x, p=2).pow(2)
            # NOTE: add nan_to_num to support batch-size==1 case.
            # not a very practically relevant case but the tests as copy-pasted seem to demand it
            return sq_pdist.mul(-self.t).exp().mean().log().nan_to_num()
        return lalign(x, y) + self.lam * (lunif(x) + lunif(y)) / 2
