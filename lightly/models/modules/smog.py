from typing import Optional

import torch
import torch.nn as nn


class SMoG(nn.Module):
    """TODO: write docstring
    
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
        """TODO: write docstring
        
        """
        n_groups, dim = new_group_features.shape
        if n_groups != self.n_groups:
            # TODO: write a nice message
            raise ValueError
        if dim != self.dim:
            # TODO: write a nice message
            raise ValueError
        self.group_features = self.new_group_features

    def update_groups(self, x: torch.Tensor):
        """TODO: write docstring
        
        """
        assignments = self.assign_groups(x)

        self.group_features = self.beta * self.group_features.detach()
        factor = (1 - self.beta)

        bincount = torch.bincount(assignments)

        for index, xi in zip(assignments, x):
            self.group_features[index] += factor * xi / bincount[index]

        return self.group_features

    @torch.no_grad()
    def assign_groups(self, x: torch.Tensor):
        """TODO: write docstring
        
        """
        x = torch.nn.functional.normalize(x)
        group_features = torch.nn.functional.normalize(self.group_features)
        logits = torch.mm(x, group_features.t())
        return torch.argmax(logits, dim=-1)
