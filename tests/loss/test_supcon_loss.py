import pytest
import torch
from torch import Tensor

from lightly.loss import NTXentLoss, SupConLoss


class TestSupConLoss:
    def test_simple_input(self) -> None:
        my_input = torch.rand([3, 2, 4])
        my_label = Tensor([[1, 0], [0, 1], [0, 1]])
        my_loss = SupConLoss()
        my_loss(my_input, my_label)

    def test_unsup_equal_to_simclr(self) -> None:
        supcon = SupConLoss(temperature=0.5, gather_distributed=False)
        ntxent = NTXentLoss(
            temperature=0.5, memory_bank_size=0, gather_distributed=False
        )
        features = torch.rand((8, 2, 10))
        supcon_loss = supcon(features)
        ntxent_loss = ntxent(features[:, 0, :], features[:, 1, :])
        assert (supcon_loss - ntxent_loss).pow(2).item() == pytest.approx(0.0)
