import unittest
from unittest import TestCase

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import SGD

from lightly.loss import msn_loss
from lightly.loss.msn_loss import MSNLoss
from lightly.models.modules.heads import MSNProjectionHead


class TestMSNLoss(TestCase):
    def test__init__temperature(self) -> None:
        MSNLoss(temperature=1.0)
        with self.assertRaises(ValueError):
            MSNLoss(temperature=0.0)
        with self.assertRaises(ValueError):
            MSNLoss(temperature=-1.0)

    def test__init__sinkhorn_iterations(self) -> None:
        MSNLoss(sinkhorn_iterations=0)
        with self.assertRaises(ValueError):
            MSNLoss(sinkhorn_iterations=-1)

    def test__init__target_distribution(self) -> None:
        MSNLoss(target_distribution="uniform")
        MSNLoss(target_distribution="power_law")
        MSNLoss(target_distribution=lambda t: t.new_ones())
        with self.assertRaises(ValueError):
            MSNLoss(target_distribution="linear")

    def test__init__power_law_exponent(self) -> None:
        MSNLoss(target_distribution="power_law", power_law_exponent=0.5)
        with self.assertRaises(ValueError):
            MSNLoss(target_distribution="power_law", power_law_exponent=0.0)
        with self.assertRaises(ValueError):
            MSNLoss(target_distribution="power_law", power_law_exponent=-1.0)

    def test__init__me_max_weight(self) -> None:
        criterion = MSNLoss(me_max_weight=0.5)
        assert criterion.regularization_weight == 0.5

        with self.assertRaises(ValueError):
            MSNLoss(target_distribution="power_law", me_max_weight=0.5)

    def test_prototype_probabilitiy(self, seed=0) -> None:
        torch.manual_seed(seed)
        queries = F.normalize(torch.rand((8, 10)), dim=1)
        prototypes = F.normalize(torch.rand((4, 10)), dim=1)
        prob = msn_loss.prototype_probabilities(queries, prototypes, temperature=0.5)
        self.assertEqual(prob.shape, (8, 4))
        self.assertLessEqual(prob.max(), 1.0)
        self.assertGreater(prob.min(), 0.0)

        # verify sharpening
        prob1 = msn_loss.prototype_probabilities(queries, prototypes, temperature=0.1)
        # same prototypes should be assigned regardless of temperature
        self.assertTrue(torch.all(prob.argmax(dim=1) == prob1.argmax(dim=1)))
        # probabilities of selected prototypes should be higher for lower temperature
        self.assertTrue(torch.all(prob.max(dim=1)[0] < prob1.max(dim=1)[0]))

    def test_sharpen(self, seed=0) -> None:
        torch.manual_seed(seed)
        prob = torch.rand((8, 10))
        p0 = msn_loss.sharpen(prob, temperature=0.5)
        p1 = msn_loss.sharpen(prob, temperature=0.1)
        # indices of max probabilities should be the same regardless of temperature
        self.assertTrue(torch.all(p0.argmax(dim=1) == p1.argmax(dim=1)))
        # max probabilities should be higher for lower temperature
        self.assertTrue(torch.all(p0.max(dim=1)[0] < p1.max(dim=1)[0]))

    def test_sinkhorn(self, seed=0) -> None:
        torch.manual_seed(seed)
        prob = torch.rand((8, 10))
        out = msn_loss.sinkhorn(prob)
        self.assertTrue(torch.all(prob != out))

    def test_sinkhorn_no_iter(self, seed=0) -> None:
        torch.manual_seed(seed)
        prob = torch.rand((8, 10))
        out = msn_loss.sinkhorn(prob, iterations=0)
        self.assertTrue(torch.all(prob == out))

    def test_forward(self, seed=0) -> None:
        torch.manual_seed(seed)
        for target_distribution in [
            "uniform",
            "power_law",
            _linear_target_distribution,
        ]:
            for num_target_views in range(1, 4):
                with self.subTest(
                    num_views=num_target_views, target_distribution=target_distribution
                ):
                    criterion = MSNLoss(target_distribution=target_distribution)
                    anchors = torch.rand((8 * num_target_views, 10))
                    targets = torch.rand((8, 10))
                    prototypes = torch.rand((4, 10), requires_grad=True)
                    criterion(anchors, targets, prototypes)

    @unittest.skipUnless(torch.cuda.is_available(), "cuda not available")
    def test_forward_cuda(self, seed=0) -> None:
        torch.manual_seed(seed)
        for target_distribution in [
            "uniform",
            "power_law",
            _linear_target_distribution,
        ]:
            with self.subTest(target_distribution=target_distribution):
                criterion = MSNLoss(target_distribution=target_distribution)
                anchors = torch.rand((8 * 2, 10)).cuda()
                targets = torch.rand((8, 10)).cuda()
                prototypes = torch.rand((4, 10), requires_grad=True).cuda()
                criterion(anchors, targets, prototypes)

    def test_backward(self, seed=0) -> None:
        torch.manual_seed(seed)
        head = MSNProjectionHead(5, 16, 6)
        for target_distribution in [
            "uniform",
            "power_law",
            _linear_target_distribution,
        ]:
            with self.subTest(target_distribution=target_distribution):
                criterion = MSNLoss(target_distribution=target_distribution)
                optimizer = SGD(head.parameters(), lr=0.1)
                anchors = torch.rand((8 * 4, 5))
                targets = torch.rand((8, 5))
                prototypes = nn.Linear(6, 4).weight  # 4 prototypes with dim 6
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
    def test_backward_cuda(self, seed=0) -> None:
        torch.manual_seed(seed)
        head = MSNProjectionHead(5, 16, 6)
        head.to("cuda")
        for target_distribution in [
            "uniform",
            "power_law",
            _linear_target_distribution,
        ]:
            with self.subTest(target_distribution=target_distribution):
                criterion = MSNLoss(target_distribution=target_distribution)
                optimizer = SGD(head.parameters(), lr=0.1)
                anchors = torch.rand((8 * 4, 5)).cuda()
                targets = torch.rand((8, 5)).cuda()
                prototypes = nn.Linear(6, 4).weight.cuda()  # 4 prototypes with dim 6
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


def test__power_law_distribution() -> None:
    power_dist = msn_loss._power_law_distribution(
        size=4, exponent=0.5, device=torch.device("cpu")
    )
    # 2.784457050376173 == sum(1/(k**0.5) for k in range(1, 5))
    assert torch.allclose(
        power_dist,
        torch.Tensor(
            [
                1 / 2.784457050376173,
                1 / (2**0.5) / 2.784457050376173,
                1 / (3**0.5) / 2.784457050376173,
                1 / (4**0.5) / 2.784457050376173,
            ]
        ),
    )
    assert power_dist.device == torch.device("cpu")
    assert torch.allclose(power_dist.sum(), torch.Tensor([1.0]))


def _linear_target_distribution(mean_anchor_probs: Tensor) -> Tensor:
    linear_dist = torch.arange(
        start=1, end=mean_anchor_probs.shape[0] + 1, device=mean_anchor_probs.device
    )
    return linear_dist / linear_dist.sum()
