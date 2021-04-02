import unittest
import torch

from lightly.loss.hypersphere_loss import HypersphereLoss


class TestHyperSphereLoss(unittest.TestCase):

    def test_forward_pass(self):
        loss = HypersphereLoss()
        for bsz in range(1, 20):

            batch_1 = torch.randn((bsz, 32))
            batch_2 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)

            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.)
