from py import log
import torch
from torch import Tensor
from torch import distributed as torch_dist
from torch.nn import Module
from lightly.utils import dist
import torch.nn.functional as F
import math

class DetConLoss(Module):
    """Implementation of the DetCon loss. [0]_

    The inputs are two views of feature maps :math:`v_m` and :math:`v_{m'}'`, pooled over the regions
    of the segmentation mask. Those feature maps are first normalized to a norm of
    :math:`\\frac{1}{\\sqrt{\\tau}}`, where :math:`\\tau` is the temperature. The contrastive
    loss is then calculated as follows, where not only different images in the batch
    are considered as negatives, but also different regions of the same image:

    .. math::
        \\mathcal{L} = \\mathbb{E}_{(m, m')\\sim \\mathcal{M}}\\left[ - \\log \\frac{\\exp(v_m \\cdot v_{m'}')}{\\exp(v_m \\cdot v_{m'}') + \\sum_{n}\\exp (v_m \\cdot v_{m'}')} \\right]

    References:
        .. [0] DetCon https://arxiv.org/abs/2103.10957

    Attributes:
        temperature:
            The temperature :math:`\\tau` in the contrastive loss.
        gather_distributed:
            If True, the similarity matrix is gathered across all GPUs before the loss
            is calculated. Else, the loss is calculated on each GPU separately.
    """
    def __init__(self, temperature: float = 0.1, gather_distributed: bool = True):
        super().__init__()
        self.eps = 1e-8
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.eps = 1e-8

        if abs(self.temperature) < self.eps:
            raise ValueError(
                "Illegal temperature: abs({}) < 1e-8".format(self.temperature)
            )
        if self.gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(self, view0: Tensor, view1: Tensor) -> Tensor:
        """Calculates the contrastive loss.

        Args:
            view0: Masked pooled feature map of the first view (B, E, N), where E is the embedding size and N is the
                number of regions/classes of the segmentation mask.
            view1: Masked pooled feature map of the second view (B, E, N), where E is the embedding size and N is the
                number of regions/classes of the segmentation mask.

        Returns:
            A scalar tensor containing the contrastive loss.
        """
        if self.gather_distributed:
            world_size = dist.world_size()
        else:
            world_size = 1

        view0 = _normalize_with_temp(view0, self.temperature, self.eps)
        view1 = _normalize_with_temp(view1, self.temperature, self.eps)

        assert torch.isnan(view0).any() == False
        assert torch.isnan(view1).any() == False, view1

        # fold the regions/classes into the batch dimension
        b, e, n = view0.size()
        view0 = view0.permute(0, 2, 1).reshape(-1, e) #(B, E, N) -> (B*N, E)
        view1 = view1.permute(0, 2, 1).reshape(-1, e) #(B, E, N) -> (B*N, E)

        # gather distributed if necessary
        if self.gather_distributed and dist.world_size() > 1:
            view0_large = torch.cat(dist.gather(view0), dim=0) # (B*N*world_size, E)
            view1_large = torch.cat(dist.gather(view1), dim=0)
            diag_mask = dist.eye_rank(b*n, device=view0.device) # (B*N, B*N*world_size)
        else:
            view0_large = view0 # (B*N, E)
            view1_large = view1 # (B*N, E)
            diag_mask = torch.eye(b*n, device=view0.device, dtype=torch.bool) # (B*N, B*N)

        # calculate similarity matrices
        logits_00 = torch.einsum("nc,mc->nm", view0, view0_large) # (B*N, B*N*world_size)
        logits_01 = torch.einsum("nc,mc->nm", view0, view1_large)
        logits_10 = torch.einsum("nc,mc->nm", view1, view0_large)
        logits_11 = torch.einsum("nc,mc->nm", view1, view1_large)

        # Remove simliarities between same views of the same image
        logits_00 = logits_00[~diag_mask].view(b*n, -1) # (B*N, B*N*world_size - 1)
        logits_11 = logits_11[~diag_mask].view(b*n, -1)

        assert logits_00.size() == (b*n*world_size, b*n*world_size - 1)

        # concat logits
        logits_0100 = torch.cat([logits_01, logits_00], dim=1) # (B*N, 2*B*N*world_size-1)
        logits_1011 = torch.cat([logits_10, logits_11], dim=1)
        logits = torch.cat([logits_0100, logits_1011], dim=0)

        assert logits.size() == (2*b*n, 2*b*n*world_size - 1)

        # calculate labels
        labels = torch.arange(b*n, device=view0.device, dtype=torch.int64)
        if self.gather_distributed:
            labels = labels + dist.rank() * b
        labels = labels.repeat(2)

        # calculate the cross-entropy loss
        return F.cross_entropy(logits, labels)

def _normalize_with_temp(x: Tensor, temperature: float, eps: float) -> Tensor:
    """Normalize to 1/sqrt(temperature).
    
    Args:
        x: Batched feature map tensor. (B, C, N)
        temperature: Temperature for normalization.

    Returns:
        Normalized feature map tensor. (B, C, N)
    """
    assert x.dim() == 3
    print()
    normed = x / (torch.norm(x, dim=1, keepdim=True) + eps)
    return normed / math.sqrt(temperature)