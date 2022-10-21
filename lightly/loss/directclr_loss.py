from cProfile import label
import torch
import torch.nn as nn

#Adapted from https://github.com/facebookresearch/directclr/blob/main/directclr/main.py
class InfoNCELoss(nn.Module):
    """Implementation of InfoNCELoss as required for DIRECTCLR"""
    def __init__(self, dim: int, temperature: float = 0.1):
        """Parameters
        Args:
            dim:
                Dimension of subvector to be used to compute InfoNCELoss.
            temprature:
                The value used to scale logits.
            dim : Dimension of subvector to be used to compute InfoNCELoss.
            temprature: The value used to scale logits.
        """
        self.temprature = temprature
        #dimension of subvector sent to infoNCE
        self.dim = dim
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Function to normalize the tensor
        Args:
            x:
                The torch tensor to be normalized.
        """
        return nn.functional.normalize(x, dim = 1)

    def compute_loss(self, z1:torch.Tensor, z2:torch.Tensor) -> torch.Tensor:
        """Method to compute InfoNCELoss
        Args:
            z1,z2:
                The representations from the encoder.
        """
        z1 = self.normalize(z1)
        z2 = self.normalize(z2)
        #DDP step
        logits = z1 @ z2.T
        logits = logits/self.temprature
        labels = torch.arange(0, z2.shape[0]).type_as(logits)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss
    
    def forward(self, z1:torch.Tensor, z2:torch.Tensor) -> torch.Tensor:
        """Forward Pass for InfoNCE computation"""
        z1 = z1[:, :self.dim]
        z2 = z2[:, :self.dim]
        loss =  self.compute_loss(z1, z2) + self.compute_loss(z2, z1)
        return loss / 2

__all__ = ["InfoNCELoss"]