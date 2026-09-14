"""The NT-Xent equation, with no state and no distributed communication."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

__all__ = ["ntxent"]


def ntxent(
    z0: Tensor,
    z1: Tensor,
    *,
    temperature: float,
    negatives: Optional[Tensor] = None,
    z0_all: Optional[Tensor] = None,
    z1_all: Optional[Tensor] = None,
    positives_at: int = 0,
) -> Tensor:
    """Normalized temperature-scaled cross entropy, as in SimCLR.

    Every argument is a tensor or a number, so this function communicates with
    no other rank and holds nothing between calls. Gathering the negatives is
    :class:`~lightly.loss.ntx_ent_loss.NTXentLoss`'s job, and it passes the
    result in through ``z0_all`` and ``z1_all``.

    Reference:
        SimCLR, 2020, https://arxiv.org/abs/2002.05709

    Args:
        z0:
            Projections of the first view, shape ``(B, D)``, L2-normalized.
        z1:
            Projections of the second view, shape ``(B, D)``, L2-normalized.
        temperature:
            Scales the logits by its inverse.
        negatives:
            Negatives from a memory bank, shape ``(D, K)``. When given, the other
            samples in the batch are not used as negatives.
        z0_all:
            The first view's projections across all ranks, shape ``(B * W, D)``.
            Defaults to ``z0``, which is the single-rank case.
        z1_all:
            The second view's projections across all ranks. Defaults to ``z1``.
        positives_at:
            The column at which this rank's rows start inside ``z0_all``, so
            ``rank * B``. Zero on one rank.

    Returns:
        The loss, as a scalar.
    """
    device = z0.device
    batch_size = z0.size(0)

    if negatives is not None:
        # sim_pos[i] is the similarity of sample i to its positive pair, and
        # sim_neg[i, j] its similarity to the j-th vector in the bank.
        sim_pos = torch.einsum("nc,nc->n", z0, z1).unsqueeze(-1)
        sim_neg = torch.einsum("nc,ck->nk", z0, negatives.to(device))
        logits = torch.cat([sim_pos, sim_neg], dim=1) / temperature
        labels = torch.zeros(logits.size(0), device=device, dtype=torch.long)
        return cross_entropy(logits, labels)

    z0_all = z0 if z0_all is None else z0_all
    z1_all = z1 if z1_all is None else z1_all

    # The similarities of a view with itself, which carry no signal and are
    # dropped below. On one rank this is torch.eye.
    rows = torch.arange(batch_size, device=device, dtype=torch.long)
    diagonal = torch.zeros(batch_size, z0_all.size(0), device=device, dtype=torch.bool)
    diagonal[rows, rows + positives_at] = True

    # n is the local batch size and m the batch across ranks, so every block is
    # (n, m).
    logits_00 = torch.einsum("nc,mc->nm", z0, z0_all) / temperature
    logits_01 = torch.einsum("nc,mc->nm", z0, z1_all) / temperature
    logits_10 = torch.einsum("nc,mc->nm", z1, z0_all) / temperature
    logits_11 = torch.einsum("nc,mc->nm", z1, z1_all) / temperature

    logits_00 = logits_00[~diagonal].view(batch_size, -1)
    logits_11 = logits_11[~diagonal].view(batch_size, -1)

    logits = torch.cat(
        [
            torch.cat([logits_01, logits_00], dim=1),
            torch.cat([logits_10, logits_11], dim=1),
        ],
        dim=0,
    )
    labels = (rows + positives_at).repeat(2)
    return cross_entropy(logits, labels)
