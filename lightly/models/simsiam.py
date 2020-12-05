import torch
import torch.nn as nn

from lightly.models.resnet import ResNetGenerator
from lightly.models.batchnorm import get_norm_layer
from lightly.models._loader import _StateDictLoaderMixin


def _prediction_mlp(in_dims: int = 2048, 
                    h_dims: int = 512, 
                    out_dims: int = 2048) -> nn.Sequential:
    """Prediction MLP. The original paper's implementation has 2 layers, with 
    BN applied to its hidden fc layers but no BN or ReLU on the output fc layer.

    Note that the hidden dimensions should be smaller than the input/output 
    dimensions (bottleneck structure). The default implementation using a 
    ResNet50 backbone has an input dimension of 2048, hidden dimension of 512, 
    and output dimension of 2048

    Args:
        in_dims:
            Input dimension of the first linear layer.
        h_dims: 
            Hidden dimension of all the fully connected layers (should be a
            bottleneck!)
        out_dims: 
            Output Dimension of the final linear layer.

    Returns:
        nn.Sequential:
            The projection head.
    """
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Linear(h_dims, out_dims)

    prediction = nn.Sequential(l1, l2)
    return prediction


def _projection_mlp(in_dims: int = 2048, 
                    h_dims: int = 2048, 
                    out_dims: int = 2048) -> nn.Sequential:
    """Projection MLP. The original paper's implementation has 3 layers, with 
    BN applied to its hidden fc layers but no ReLU on the output fc layer.

    Args:
        in_dims:
            Input dimension of the first linear layer.
        h_dims: 
            Hidden dimension of all the fully connected layers.
        out_dims: 
            Output Dimension of the final linear layer.

    Returns:
        nn.Sequential:
            The projection head.
    """
    l1 = nn.Sequential(nn.Linear(in_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l2 = nn.Sequential(nn.Linear(h_dims, h_dims),
                       nn.BatchNorm1d(h_dims),
                       nn.ReLU(inplace=True))

    l3 = nn.Sequential(nn.Linear(h_dims, out_dims),
                       nn.BatchNorm1d(out_dims))

    projection = nn.Sequential(l1, l2, l3)
    return projection


class SimSiam(nn.Module, _StateDictLoaderMixin):
    """ Implementation of SimSiam network

    Attributes:
        backbone:
            TODO
        width:
            Width of the ResNet.
        num_ftrs:
            Dimension of the embedding (before the projection head).
        proj_hidden_dim:
            TODO
        pred_hidden_dim:
            TODO
        out_dim:
            Dimension of the output (after the projection head).

    """

    def __init__(self,
                 backbone: nn.Module,
                 num_ftrs: int = 2048,
                 proj_hidden_dim: int = 2048,
                 pred_hidden_dim: int = 512,
                 out_dim: int = 2048):

        super(SimSiam, self).__init__()

        self.backbone = backbone
        self.num_ftrs = num_ftrs
        self.proj_hidden_dim = proj_hidden_dim
        self.pred_hidden_dim = pred_hidden_dim
        self.out_dim = out_dim

        self.projection_mlp = \
            _projection_mlp(num_ftrs, proj_hidden_dim, out_dim)

        self.prediction_mlp = \
            _prediction_mlp(num_ftrs, pred_hidden_dim, out_dim)
        
    def load_from_state_dict(self,
                             state_dict,
                             strict: bool = True,
                             apply_filter: bool = True):
        """Initializes a ResNetMoCo and loads weights from a checkpoint.

        Args:
            state_dict:
                State dictionary with layer weights.
            strict:
                Set to False when loading from a partial state_dict.
            apply_filter:
                If True, removes the `model.` prefix from keys in the state_dict.

        """
        self._custom_load_from_state_dict(state_dict, strict, apply_filter)

    def forward(self, x: torch.Tensor):
        """Forward pass through SimSiam.

        Extracts features with the backbone and apply the projection
        head to the output space.

        Args:
            x:
                Tensor of shape bsz x channels x W x H

        Returns:
            Tensor of shape bsz x out_dim

        """
        # device = output.device
        batch_size = len(x)
        batch_size = batch_size // 2

        x1, x2 = x[:batch_size], x[batch_size:]

        emb1, emb2 = self.backbone(x1), self.backbone(x2)
        emb1, emb2 = emb1.squeeze(), emb2.squeeze()
        z1, z2 = self.projection_mlp(emb1), self.projection_mlp(emb2)
        p1, p2 = self.prediction_mlp(z1), self.prediction_mlp(z2)

        output = torch.cat((z1, z2, p1, p2), 0)
        return output
