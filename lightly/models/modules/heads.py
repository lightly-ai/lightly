# TODO

from typing import Union, List, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d


class ProjectionHead(nn.Module):
    """TODO
    
    """

    def __init__(self, blocks: List[Tuple[int, int, nn.Module, nn.Module]]):

        super(ProjectionHead, self).__init__()

        self.layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            self.layers.append(nn.Linear(input_dim, output_dim))
            if batch_norm:
                self.layers.append(batch_norm)
            if non_linearity:
                self.layers.append(non_linearity)
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        """TODO
        
        """
        return self.layers(x)

    @property
    def in_features(self):
        # the first layer is always a linear layer
        return self.layers[-1].in_features

    @property
    def output_dim(self):
        return self.layers[-1].out_features

class BarlowTwinsProjectionHead(ProjectionHead):
    """TODO
    
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        """TODO
        
        """
        super(BarlowTwinsProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class BYOLProjectionHead(ProjectionHead):
    """TODO
    
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        """TODO
        
        """
        super(BarlowTwinsProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class MoCoProjectionHead(ProjectionHead):
    """TODO
    
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(MoCoProjectionHead, self).__init__([
            (input_dim, input_dim, None, nn.ReLU()),
            (input_dim, output_dim, None, None),
        ])


class NNCLRProjectionHead(ProjectionHead):
    """TODO
    
    """
    pass


class NNCLRPredictionHead(ProjectionHead):
    """TODO
    
    """
    pass


class SimCLRProjectionHead(ProjectionHead):
    """TODO
    
    """
    pass


class SimSiamProjectionHead(ProjectionHead):
    """TODO
    
    """
    pass


class SimSiamPredictionHead(ProjectionHead):
    """TODO
    
    """
    pass