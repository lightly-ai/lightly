from importlib.metadata import requires
from typing import Optional

import torch
import torch.nn as nn


class SMoG(nn.Module):
    """SMoG module for synchronous momentum grouping.
    
    """

    def __init__(
        self, group_features: torch.Tensor, beta: float,
    ):
        super(SMoG, self).__init__()
        self.group_features = group_features
        self.beta = beta

    def forward(self, x: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """TODO"""
        x = torch.nn.functional.normalize(x, dim=1)
        group_features = torch.nn.functional.normalize(self.group_features, dim=1)
        logits = torch.mm(x, group_features.t())
        return logits / temperature

    def update_groups(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the synchronous momentum update of the group vectors.

        Args:
            x:
                Tensor of shape bsz x dim.

        Returns:
            The update group features.

        """
        assignments = self.assign_groups(x)
        # bincount = torch.bincount(
        #     assignments,
        #     minlength=self.group_features.shape[0]
        # )
        # mask = torch.nonzero(bincount)
        # self.group_features = self.group_features.detach()
        # self.group_features[mask] *= self.beta

        # factor = (1 - self.beta)
        # for index, xi in zip(assignments, x):
        #     self.group_features[index] += factor * xi / bincount[index]
        self.group_features = self.group_features.detach()
        for assigned_class in torch.unique(assignments): 
            mask = assignments == assigned_class
            self.group_features[assigned_class] = self.beta * self.group_features[assigned_class] + (1 - self.beta) * x[mask].mean(axis=0)

        self.group_features= nn.functional.normalize(self.group_features, dim=1)

    @torch.no_grad()
    def assign_groups(self, x: torch.Tensor) -> torch.LongTensor:
        """Assigns each representation in x to a group based on cosine similarity.

        Args:
            Tensor of shape bsz x dim.

        Returns:
            LongTensor of shape bsz indicating group assignments.
        
        """
        return torch.argmax(self.forward(x), dim=-1)
