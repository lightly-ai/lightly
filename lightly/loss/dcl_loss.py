from typing import Callable, Optional
from functools import partial

import torch
from torch import Tensor

from lightly.utils import dist

def negative_mises_fisher_weights(out0: Tensor, out1: Tensor, sigma: float=0.5):
    similarity = torch.einsum('nm,nm->n', out0, out1) / sigma
    return 2 - out0.shape[0] * torch.nn.functional.softmax(similarity, dim=0)

class DCLLoss(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        weight_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        gather_distributed: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn
        self.gather_distributed = gather_distributed

    def forward(
        self,
        out0: Tensor,
        out1: Tensor,
    ) -> Tensor:
        # normalize the output to length 1
        out0 = out0_all = torch.nn.functional.normalize(out0, dim=1)
        out1 = out1_all = torch.nn.functional.normalize(out1, dim=1)

        if self.gather_distributed and dist.world_size() > 1:
            # gather representations from other processes if necessary
            out0_all = torch.cat(dist.gather(out0), 0)
            out1_all = torch.cat(dist.gather(out1), 0)

        # calculate symmetric loss
        loss0 = self._loss(out0, out1, out0_all, out1_all)
        loss1 = self._loss(out1, out0, out1_all, out0_all)
        return 0.5 * (loss0 + loss1)

    def _loss(self, out0, out1, out0_all, out1_all):
        # create diagonal mask that only selects similarities between
        # representations of the same images
        batch_size = out0.shape[0]
        if self.gather_distributed and dist.world_size() > 1:
            diag_mask = dist.eye_rank(batch_size, device=out0.device)
        else:
            diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)

        # calculate similarities
        # here n = batch_size and m = batch_size * world_size.
        sim_00 = torch.einsum('nc,mc->nm', out0, out0_all) / self.temperature
        sim_01 = torch.einsum('nc,mc->nm', out0, out1_all) / self.temperature

        positive_loss = -sim_01[diag_mask]
        if self.weight_fn:
            positive_loss = positive_loss * self.weight_fn(out0, out1)

        # remove simliarities between same views of the same image
        sim_00 = sim_00[~diag_mask].view(batch_size, -1)
        # remove similarities between different views of the same images
        # this is the key difference compared to NTXentLoss
        sim_01 = sim_01[~diag_mask].view(batch_size, -1)

        negative_loss_00 = torch.logsumexp(sim_00, dim=1)
        negative_loss_01 = torch.logsumexp(sim_01, dim=1)
        return (positive_loss + negative_loss_00 + negative_loss_01).mean()

class DCLWLoss(DCLLoss):
    def __init__(
        self,
        temperature: float = 0.1,
        sigma: float = 0.5,
        gather_distributed: bool = False,
    ):
        super().__init__(
            temperature=temperature,
            weight_fn=partial(negative_mises_fisher_weights, sigma=sigma),
            gather_distributed=gather_distributed,
        )
