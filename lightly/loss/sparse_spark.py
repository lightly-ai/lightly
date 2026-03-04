from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, nn

from lightly.utils.dist import gather


class SparKPatchReconLoss(nn.Module):
    """Computes per-patch normalized reconstruction loss for masked regions.

    Calculates L2 loss between reconstructed and original patches, normalized per-patch
    to account for varying feature statistics. Loss is computed only on masked (inactive) regions.

    Args:
        eps: Small value for numerical stability. Default: 1e-6.
    """

    def __init__(self, eps: float = 1e-6, gather_distributed: bool = False) -> None:
        super().__init__()
        if gather_distributed and not dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )
        self.eps = eps
        self.gather_distributed = gather_distributed

    def forward(
        self,
        inp_patches: Tensor,
        rec_patches: Tensor,
        active_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute reconstruction loss and per-patch statistics.

        Normalizes original patches based on per-patch mean and variance, then computes
        L2 loss between normalized original and reconstructed patches. Averages loss
        only over masked (active_mask=False) patches.

        Args:
            inp_patches: Original patches of shape (B, L, N) where B=batch, L=length, N=patch_dim.
            rec_patches: Reconstructed patches of shape (B, L, N).
            active_mask: Boolean mask of shape (B, 1, f, f) indicating active regions.
                        Must have 4 dimensions (2D spatial mask).

        Returns:
            Tuple of:
            - recon_loss: Scalar tensor with averaged reconstruction loss on masked regions.
            - mean: Per-patch mean of shape (B, L, 1).
            - var: Per-patch standard deviation of shape (B, L, 1).

        Raises:
            ValueError: If active_mask does not have 4 dimensions.
        """
        if active_mask.ndim != 4:
            raise ValueError(
                "active_mask must be non-flattened with shape (B, 1, f, f)"
            )

        mean = inp_patches.mean(dim=-1, keepdim=True)
        var = (inp_patches.var(dim=-1, keepdim=True) + self.eps) ** 0.5

        inp_normalized = (inp_patches - mean) / var

        l2_loss = ((rec_patches - inp_normalized) ** 2).mean(dim=2)

        non_active = active_mask.logical_not().int().view(active_mask.shape[0], -1)

        local_numerator = (l2_loss * non_active).sum()
        local_denominator = non_active.sum()

        if self.gather_distributed and dist.is_available() and dist.is_initialized():
            global_numerator = torch.cat(
                gather(local_numerator.unsqueeze(0)), dim=0
            ).sum()
            global_denominator = torch.cat(
                gather(local_denominator.unsqueeze(0)), dim=0
            ).sum()
        else:
            global_numerator = local_numerator
            global_denominator = local_denominator

        recon_loss = global_numerator / (global_denominator + self.eps)
        return recon_loss, mean, var
