"""
FIXME: hypersphere is perhaps bad naming as I am not sure it is the essence;
 alignment-and-uniformity loss perhaps? Does not sound as nice.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class HypersphereLoss(Module):
    """Implementation of the loss described in 'Understanding Contrastive Representation Learning through
    Alignment and Uniformity on the Hypersphere.' [0]

    [0] Tongzhou Wang. et.al, 2020, ... https://arxiv.org/abs/2005.10242

    Note:
        In order for this loss to function as advertized, an L1-normalization to the hypersphere is required.
        This loss function applies this L1-normalization internally in the loss layer.
        However, it is recommended that the same normalization is also applied in your architecture,
        considering that this L1-loss is also intended to be applied during inference.
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

    def __init__(self, t: float = 1.0, lam: float = 1.0, alpha: float = 2.0):
        """Initializes the HypersphereLoss module with the specified parameters.

        Parameters as described in [0]

        Args:
            t:
                Temperature parameter; proportional to the inverse variance of the Gaussians used to measure uniformity.
            lam:
                Weight balancing the alignment and uniformity loss terms
            alpha:
                Power applied to the alignment term of the loss. At its default value of 2,
                distances between positive samples are penalized in an L2 sense.
        """
        super(HypersphereLoss, self).__init__()
        self.t = t
        self.lam = lam
        self.alpha = alpha

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        """Computes the Hypersphere loss, which combines alignment and uniformity loss terms.

        Args:
            z_a:
                Tensor of shape (batch_size, embedding_dim) for the first set of embeddings.
            z_b:
                Tensor of shape (batch_size, embedding_dim) for the second set of embeddings.

        Returns:
            The computed loss.
        """
        # Normalize the input embeddings
        x = F.normalize(z_a)
        y = F.normalize(z_b)

        # Calculate alignment loss
        def lalign(x: Tensor, y: Tensor) -> Tensor:
            lalign_: Tensor = (x - y).norm(dim=1).pow(self.alpha).mean()
            return lalign_

        # Calculate uniformity loss
        def lunif(x: Tensor) -> Tensor:
            sq_pdist = torch.pdist(x, p=2).pow(2)
            return sq_pdist.mul(-self.t).exp().mean().log()

        # Combine alignment and uniformity loss terms
        return lalign(x, y) + self.lam * (lunif(x) + lunif(y)) / 2.0
