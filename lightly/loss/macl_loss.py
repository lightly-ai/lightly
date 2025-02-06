import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MACLLoss(nn.Module):
    """Implementation of the Model-Aware Contrastive Loss (MACL) from the paper:

    This implementation follows the MACL[0] paper.

    - [0] Model-Aware Contrastive Learning: Towards Escaping the Dilemmas, ICML 2023, https://arxiv.org/abs/2207.07874

    Attributes:
        temperature: Base temperature. Range: (0, inf)
        alpha: Scaling factor for controlling how much the temperature changes. Range: [0, 1]
        A_0: Initial threshold for the alignment magnitude. Range: [0, 1]

    Raises:
        ValueError:
            If the initial temperature is less than 1e-8.
            If the alpha value is not in the range [0, 1].
            If the A_0 value is not in the range [0, 1].

    Examples:
        >>> # initialize the loss function
        >>> loss_fn = MACLLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimCLR or MoCo model
        >>> z0 = model(t0)
        >>> z1 = model(t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(z0, z1)
    """

    def __init__(self, temperature: float = 0.1, alpha: float = 0.5, A_0: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.A_0 = A_0
        self.eps = 1e-8

        if self.temperature < self.eps:
            raise ValueError(
                f"Illegal initial temperature: abs({self.temperature}) < 1e-8"
            )

        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("Alpha must be in the range [0, 1].")

        if self.A_0 < 0 or self.A_0 > 1:
            raise ValueError("A_0 must be in the range [0, 1].")

    def forward(self, z0: Tensor, z1: Tensor) -> Tensor:
        """Compute the Model-Aware Contrastive Loss (MACL) for a batch of embeddings.

        Args:
            z0:
                First view embeddings
                Shape (batch_size, embedding_size)
            z1:
                Second view embeddings
                Shape (batch_size, embedding_size)

        Returns:
            loss:
                The computed loss value
        """
        # Normalize embeddings
        z0 = F.normalize(z0, dim=-1, p=2)
        z1 = F.normalize(z1, dim=-1, p=2)

        # Concatenate embeddings
        out = torch.cat([z0, z1], dim=0)
        batch_size = z0.shape[0]
        n_samples = len(out)

        # Compute similarity matrix
        cov = out @ out.T

        # Get positive and negative pairs
        mask = cov.new_ones(cov.shape, dtype=torch.bool)
        mask.diagonal()[:] = False
        mask.diagonal(batch_size)[:] = False
        mask.diagonal(-batch_size)[:] = False
        neg = cov.masked_select(mask).view(n_samples, -1)

        # Get positive pairs from upper and lower diagonals
        u_b = torch.diag(cov, batch_size)
        l_b = torch.diag(cov, -batch_size)
        pos = torch.cat([u_b, l_b], dim=0).reshape(n_samples, 1)

        # Calculate model-aware temperature
        A = torch.mean(pos.detach())
        t = self.temperature * (1 + self.alpha * (A - self.A_0))

        # 6) Compute stable log_softmax
        logits = torch.cat([pos, neg], dim=1) / t
        log_prob = F.log_softmax(logits, dim=1)

        # 7) Extract log probability of the positive pair
        log_pos_prob = log_prob[:, 0]

        # 8) Compute P and V
        P = log_pos_prob.exp()
        V = 1.0 / (1.0 - P + self.eps)  # add eps to avoid division-by-zero

        # 9) Compute final loss
        loss: Tensor = -V.detach() * log_pos_prob

        return loss.mean()
