import unittest
import torch
from torch import nn

from lightly.utils import cosine_schedule
from lightly.utils import CosineWarmupScheduler


class TestModelUtils(unittest.TestCase):
    def test_cosine_schedule(self):
        momentum_0 = cosine_schedule(1, 10, 0.99, 1)
        momentum_hand_computed_0 = 0.99030154
        momentum_1 = cosine_schedule(95, 100, 0.7, 2)
        momentum_hand_computed_1 = 1.99477063
        momentum_2 = cosine_schedule(0, 1, 0.996, 1)
        momentum_hand_computed_2 = 1

        self.assertAlmostEqual(momentum_0, momentum_hand_computed_0, 6)
        self.assertAlmostEqual(momentum_1, momentum_hand_computed_1, 6)
        self.assertAlmostEqual(momentum_2, momentum_hand_computed_2, 6)

    def test_CosineWarmupScheduler(self):
        model = nn.Linear(10, 1)
        optim = torch.optim.SGD(
            model.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        warmup_epochs = 3
        max_epochs = 10
        scheduler = CosineWarmupScheduler(optim, warmup_epochs, max_epochs)
