# TODO

from typing import Union, List, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d


class ProjectionHead(nn.Module):
    """TODO
    
    """

    def __init__(self, layers: List[Tuple[int, int, nn.Module, nn.Module]]):

        self.ffnn = []
        for input_dim, output_dim, batch_norm, non_linearity in layers:
            self.ffnn.append(nn.Linear(input_dim, output_dim))
            if batch_norm:
                self.ffnn.append(batch_norm)
            if non_linearity:
                self.ffnn.append(non_linearity)
        self.ffnn = nn.Sequential(*self.ffnn)

        # TODO
        self.input_dim = layers[0][0]
        self.output_dim = layers[-1][1]

    def forward(self, x: torch.Tensor):
        """TODO
        
        """
        return self.ffnn(x)


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