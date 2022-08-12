from typing import Optional

import torch
import torch.nn as nn


class SMoG(nn.Module):
    """SMoG module for synchronous momentum grouping.

    Attributes:
        n_groups:
            Number of groups to cluster.
        dim:
            Dimensionality of the group representations.
        beta:
            Momentum update factor of the group representations.
        device:
            Device to select.
    
    """

    def __init__(
        self, n_groups: int, dim: int, beta: float, device: Optional[str] = None
    ):
        super(SMoG, self).__init__()
        self.n_groups = n_groups
        self.dim = dim
        self.group_features = torch.rand(n_groups, dim).to(device)
        self.beta = beta

    def init_groups(self, new_group_features: torch.Tensor):
        """Replaces the current group features by a new set of group features.

        Args:
            Tensor of shape n_groups x dim.
        
        """
        n_groups, dim = new_group_features.shape
        if n_groups != self.n_groups:
            # TODO: write a nice message
            raise ValueError
        if dim != self.dim:
            # TODO: write a nice message
            raise ValueError
        self.group_features = new_group_features

    def update_groups(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the synchronous momentum update of the group vectors.

        Args:
            x:
                Tensor of shape bsz x dim.

        Returns:
            The update group features.

        """
        assignments = self.assign_groups(x)
        bincount = torch.bincount(assignments)
        mask = torch.nonzero(bincount)
        self.group_features = self.group_features.detach()
        self.group_features[mask] *= self.beta

        factor = (1 - self.beta)
        for index, xi in zip(assignments, x):
            self.group_features[index] += factor * xi / bincount[index]

        self.group_features = nn.functional.normalize(self.group_features)
        return self.group_features

    @torch.no_grad()
    def assign_groups(self, x: torch.Tensor) -> torch.LongTensor:
        """Assigns each representation in x to a group based on cosine similarity.

        Args:
            Tensor of shape bsz x dim.

        Returns:
            LongTensor of shape bsz indicating group assignments.
        
        """
        x = torch.nn.functional.normalize(x)
        group_features = torch.nn.functional.normalize(self.group_features)
        logits = torch.mm(x, group_features.t())
        return torch.argmax(logits, dim=-1)
