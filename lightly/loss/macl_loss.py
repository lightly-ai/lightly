import torch
import torch.nn as nn
import torch.nn.functional as F

class MACLLoss(nn.Module):
    """Implementation of the Model-Aware Contrastive Loss (MACL) from the paper:
    
    This implementation follows the MACL[0] paper.

    - [0] Model-Aware Contrastive Learning: Towards Escaping the Dilemmas, ICML 2023, https://arxiv.org/abs/2207.07874

    Attributes:
        t_0: Base temperature
        alpha: Scaling factor for controlling how much the temperature changes. Range: [0, 1]
        A_0: Initial threshold for the alignment magnitude.

    Raises:
        ValueError: If the alpha value is not in the range [0, 1].

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
    
    def __init__(self, t_0: float = 0.1, alpha: float = 0.5, A_0: float = 0.0):
        super().__init__()
        self.t_0 = t_0
        self.alpha = alpha
        self.A_0 = A_0
        
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be in the range [0, 1].")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        
        return mask

    def forward(self, z0, z1):
        """ Compute the Model-Aware Contrastive Loss (MACL) for a batch of embeddings.
        
        Args:
            z0: 
                First view embeddings
                shape (batch_size, embedding_size)
            z1: 
                Second view embeddings
                Shape (batch_size, embedding_size)
            
        Returns:
            loss: MACL loss
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
        mask = cov.new_ones(cov.shape, dtype=bool)
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
        t = self.t_0 * (1 + self.alpha * (A - self.A_0))
        
        # Calculate logits and loss
        logits = torch.cat([pos, neg], dim=1)
        P = torch.softmax(logits/t, dim=1)[:, 0]
        V = 1 / (1 - P)
        loss = -V.detach() * torch.log(P)
        
        return loss.mean()
