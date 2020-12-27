import unittest
import torch

from lightly.loss import SymNegCosineSimilarityLoss


class TestSymNegCosineSimilarityLoss(unittest.TestCase):
    def test_forward_pass(self):
        loss = SymNegCosineSimilarityLoss()
        for bsz in range(1, 20):

            z0 = torch.randn((bsz, 32))
            p0 = torch.randn((bsz, 32))
            z1 = torch.randn((bsz, 32))
            p1 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss((z0, p0), (z1, p1))
            l2 = loss((z1, p1), (z0, p0))
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.)
 

    def test_forward_pass_cuda(self):
        if not torch.cuda.is_available():
            return

        loss = SymNegCosineSimilarityLoss()
        for bsz in range(1, 20):

            z0 = torch.randn((bsz, 32)).cuda()
            p0 = torch.randn((bsz, 32)).cuda()
            z1 = torch.randn((bsz, 32)).cuda()
            p1 = torch.randn((bsz, 32)).cuda()

            # symmetry
            l1 = loss((z0, p0), (z1, p1))
            l2 = loss((z1, p1), (z0, p0))
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.)


    def test_neg_cosine_simililarity(self):
        loss = SymNegCosineSimilarityLoss()
        for bsz in range(1, 20):

            x = torch.randn((bsz, 32))
            y = torch.randn((bsz, 32))

            # symmetry
            l1 = loss._neg_cosine_simililarity(x, y)
            l2 = loss._neg_cosine_simililarity(y, x)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.)

    def test_neg_cosine_simililarity_cuda(self):
        if not torch.cuda.is_available():
            return

        loss = SymNegCosineSimilarityLoss()
        for bsz in range(1, 20):

            x = torch.randn((bsz, 32)).cuda()
            y = torch.randn((bsz, 32)).cuda()

            # symmetry
            l1 = loss._neg_cosine_simililarity(x, y)
            l2 = loss._neg_cosine_simililarity(y, x)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.)
