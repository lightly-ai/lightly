import unittest
import torch

from lightly.loss import NegativeCosineSimilarity


class TestNegativeCosineSimilarity(unittest.TestCase):
    def test_forward_pass(self):
        loss = NegativeCosineSimilarity()
        for bsz in range(1, 20):
            x0 = torch.randn((bsz, 32))
            x1 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss(x0, x1)
            l2 = loss(x1, x0)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available")
    def test_forward_pass_cuda(self):
        loss = NegativeCosineSimilarity()
        for bsz in range(1, 20):
            x0 = torch.randn((bsz, 32)).cuda()
            x1 = torch.randn((bsz, 32)).cuda()

            # symmetry
            l1 = loss(x0, x1)
            l2 = loss(x1, x0)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0)
