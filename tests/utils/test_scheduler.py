import unittest
import torch
from torch import nn

from lightly.utils import cosine_schedule
from lightly.utils import CosineWarmupScheduler


class TestScheduler(unittest.TestCase):
    def test_cosine_schedule(self):
        self.assertAlmostEqual(cosine_schedule(1, 10, 0.99, 1.0), 0.99030154, 6)
        self.assertAlmostEqual(cosine_schedule(95, 100, 0.7, 2.0), 1.99477063, 6)
        self.assertAlmostEqual(cosine_schedule(0, 1, 0.996, 1.0), 1.0, 6)
        self.assertAlmostEqual(cosine_schedule(10, 10, 0.0, 1.0), 1.0, 6)
        with self.assertRaises(ValueError):
            cosine_schedule(-1, 1, 0.0, 1.0)
        with self.assertRaises(ValueError):
            cosine_schedule(0, 0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            cosine_schedule(1, 0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            cosine_schedule(11, 10, 0.0, 1.0)

    def test_CosineWarmupScheduler(self):
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=1.0, momentum=0.0, weight_decay=0.0
        )
        scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=3, max_epochs=6, verbose=True)

        # warmup
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.333333333)
        scheduler.step()
        optimizer.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.666666666)
        scheduler.step()
        optimizer.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1.0)
        scheduler.step()
        optimizer.step()

        # cosine decay
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1.0)
        scheduler.step()
        optimizer.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.5)
        scheduler.step()
        optimizer.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.0)

        # extra step for Pytorch Lightning
        scheduler.step()
        optimizer.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.0)

        # step > max_epochs
        with self.assertRaises(ValueError):
            scheduler.step()
