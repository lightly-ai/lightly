import unittest
from unittest import TestCase
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD

from lightly.models.modules.heads import MSNProjectionHead
from lightly.loss import msn_loss
from lightly.loss.msn_loss import MSNLoss

class TestMSNLoss(TestCase):

    def test_prototype_probabilitiy(self, seed=0):
        torch.manual_seed(seed)
        queries = F.normalize(torch.rand((8, 10)), dim=1)
        prototypes = F.normalize(torch.rand((4, 10)), dim=1)
        prob = msn_loss.prototype_probabilities(queries, prototypes, temperature=0.5)
        self.assertEqual(prob.shape, (8, 4))
        self.assertLessEqual(prob.max(), 1.0)
        self.assertGreater(prob.min(), 0.0)

        # verify sharpening
        prob1 = msn_loss.prototype_probabilities(queries, prototypes, temperature=0.1)
        # same prototypes should be assigned regardless of temperature
        self.assertTrue(torch.all(prob.argmax(dim=1) == prob1.argmax(dim=1)))
        # probabilities of selected prototypes should be higher for lower temperature
        self.assertTrue(torch.all(prob.max(dim=1)[0] < prob1.max(dim=1)[0]))

    def test_sharpen(self, seed=0):
        torch.manual_seed(seed)
        prob = torch.rand((8, 10))
        p0 = msn_loss.sharpen(prob, temperature=0.5)
        p1 = msn_loss.sharpen(prob, temperature=0.1)
        # indices of max probabilities should be the same regardless of temperature
        self.assertTrue(torch.all(p0.argmax(dim=1) == p1.argmax(dim=1)))
        # max probabilities should be higher for lower temperature
        self.assertTrue(torch.all(p0.max(dim=1)[0] < p1.max(dim=1)[0]))

    def test_sinkhorn(self, seed=0):
        torch.manual_seed(seed)
        prob = torch.rand((8, 10))
        out = msn_loss.sinkhorn(prob)
        self.assertTrue(torch.all(prob != out))

    def test_forward(self, seed=0):
        torch.manual_seed(seed)
        criterion = MSNLoss()

        for num_target_views in range(1, 4):
            with self.subTest(num_views=num_target_views):
                anchors = torch.rand((8 * num_target_views, 10))
                targets = torch.rand((8, 10))
                prototypes = torch.rand((4, 10), requires_grad=True)
                criterion(anchors, targets, prototypes)

    @unittest.skipUnless(torch.cuda.is_available(), "cuda not available")
    def test_forward_cuda(self, seed=0):
        torch.manual_seed(seed)
        criterion = MSNLoss()
        anchors = torch.rand((8 * 2, 10)).cuda()
        targets = torch.rand((8, 10)).cuda()
        prototypes = torch.rand((4, 10), requires_grad=True).cuda()
        criterion(anchors, targets, prototypes)

    def test_backward(self, seed=0):
        torch.manual_seed(seed)
        head = MSNProjectionHead(5, 16, 6)
        criterion = MSNLoss()
        optimizer = SGD(head.parameters(), lr=0.1)
        anchors = torch.rand((8 * 4, 5))
        targets = torch.rand((8, 5))
        prototypes = nn.Linear(6, 4).weight # 4 prototypes with dim 6
        optimizer.zero_grad()
        anchors = head(anchors)
        with torch.no_grad():
            targets = head(targets)
        loss = criterion(anchors, targets, prototypes)
        loss.backward()
        weights_before = head.layers[0].weight.data.clone()
        optimizer.step()
        weights_after = head.layers[0].weight.data
        # backward pass should update weights
        self.assertTrue(torch.any(weights_before != weights_after))

    @unittest.skipUnless(torch.cuda.is_available(), "cuda not available")
    def test_backward_cuda(self, seed=0):
        torch.manual_seed(seed)
        head = MSNProjectionHead(5, 16, 6)
        head.to('cuda')
        criterion = MSNLoss()
        optimizer = SGD(head.parameters(), lr=0.1)
        anchors = torch.rand((8 * 4, 5)).cuda()
        targets = torch.rand((8, 5)).cuda()
        prototypes = nn.Linear(6, 4).weight.cuda() # 4 prototypes with dim 6
        optimizer.zero_grad()
        anchors = head(anchors)
        with torch.no_grad():
            targets = head(targets)
        loss = criterion(anchors, targets, prototypes)
        loss.backward()
        weights_before = head.layers[0].weight.data.clone()
        optimizer.step()
        weights_after = head.layers[0].weight.data
        # backward pass should update weights
        self.assertTrue(torch.any(weights_before != weights_after))
