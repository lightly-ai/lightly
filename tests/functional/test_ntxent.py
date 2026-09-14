"""Golden values for the NT-Xent equation.

The numbers below are the second statement of the equation, as data rather than
as code. Editing ``lightly/functional/ntxent.py`` moves them, and someone then
has to approve the new number or restore the old one. A pinned loss sees changes
above roughly 1e-5.
"""

from typing import Tuple

import pytest
import torch
from torch import Tensor
from torch.nn.functional import normalize

from lightly.functional import ntxent
from lightly.loss import NTXentLoss

GOLDEN_LOSS = 5.350811958313  # seed 11, batch 8, dim 16, temperature 0.1
GOLDEN_LOSS_MEMORY_BANK = 9.919633865356  # the same batch, 8 negatives from a bank
TOLERANCE = 1e-5


def batch(seed: int = 11) -> Tuple[Tensor, Tensor]:
    torch.manual_seed(seed)
    return (
        normalize(torch.randn(8, 16), dim=1),
        normalize(torch.randn(8, 16), dim=1),
    )


def test_golden_loss() -> None:
    z0, z1 = batch()
    assert float(ntxent(z0, z1, temperature=0.1)) == pytest.approx(
        GOLDEN_LOSS, abs=TOLERANCE
    )


def test_golden_loss_with_a_memory_bank() -> None:
    z0, z1 = batch()
    torch.manual_seed(11)
    negatives = normalize(torch.randn(8, 16), dim=1).T
    loss = ntxent(z0, z1, temperature=0.1, negatives=negatives)
    assert float(loss) == pytest.approx(GOLDEN_LOSS_MEMORY_BANK, abs=TOLERANCE)


def test_the_module_computes_the_same_equation() -> None:
    z0, z1 = batch()
    module = NTXentLoss(temperature=0.1)
    assert float(module(z0, z1)) == pytest.approx(GOLDEN_LOSS, abs=TOLERANCE)


def test_temperature_scales_the_logits() -> None:
    z0, z1 = batch()
    assert float(ntxent(z0, z1, temperature=0.1)) > float(
        ntxent(z0, z1, temperature=0.5)
    )


def test_positives_at_zero_is_the_single_rank_case() -> None:
    z0, z1 = batch()
    with_all = ntxent(z0, z1, temperature=0.1, z0_all=z0, z1_all=z1, positives_at=0)
    assert torch.equal(with_all, ntxent(z0, z1, temperature=0.1))


def test_the_loss_declares_how_it_behaves_across_ranks() -> None:
    from lightly.loss import DistributedKind

    assert NTXentLoss.distributed_kind is DistributedKind.GATHER_FOR_NEGATIVES
