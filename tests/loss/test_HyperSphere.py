import unittest
import torch

from lightly.loss.hypersphere_loss import HypersphereLoss


class TestHyperSphereLoss(unittest.TestCase):

    def test_forward_pass(self):
        loss = HypersphereLoss()
        # NOTE: skipping bsz==1 case as its not relevant to this loss, and will produce nan-values
        for bsz in range(2, 20):

            batch_1 = torch.randn((bsz, 32))
            batch_2 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)

            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.)
