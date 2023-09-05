import math
import unittest

import pytest
import torch
from torch import Tensor

from lightly.loss import pmsn_loss
from lightly.loss.pmsn_loss import PMSNCustomLoss, PMSNLoss


class TestPMSNLoss:
    def test_regularization_loss(self) -> None:
        criterion = PMSNLoss()
        mean_anchor_probs = torch.Tensor([0.1, 0.3, 0.6])
        loss = criterion.regularization_loss(mean_anchor_probs=mean_anchor_probs)
        norm = 1 / (1**0.25) + 1 / (2**0.25) + 1 / (3**0.25)
        t0 = 1 / (1**0.25) / norm
        t1 = 1 / (2**0.25) / norm
        t2 = 1 / (3**0.25) / norm
        loss = criterion.regularization_loss(mean_anchor_probs=mean_anchor_probs)
        expected_loss = (
            t0 * math.log(t0 / 0.1) + t1 * math.log(t1 / 0.3) + t2 * math.log(t2 / 0.6)
        )
        assert loss == pytest.approx(expected_loss)

    def test_forward(self) -> None:
        torch.manual_seed(0)
        criterion = PMSNLoss()
        anchors = torch.rand((8 * 3, 10))
        targets = torch.rand((8, 10))
        prototypes = torch.rand((4, 10), requires_grad=True)
        criterion(anchors, targets, prototypes)

    @unittest.skipUnless(torch.cuda.is_available(), "cuda not available")
    def test_forward_cuda(self) -> None:
        torch.manual_seed(0)
        criterion = PMSNLoss()
        anchors = torch.rand((8 * 3, 10)).cuda()
        targets = torch.rand((8, 10)).cuda()
        prototypes = torch.rand((4, 10), requires_grad=True).cuda()
        criterion(anchors, targets, prototypes)


class TestPMSNCustomLoss:
    def test_regularization_loss(self) -> None:
        criterion = PMSNCustomLoss(target_distribution=_uniform_distribution)
        mean_anchor_probs = torch.Tensor([0.1, 0.3, 0.6])
        loss = criterion.regularization_loss(mean_anchor_probs=mean_anchor_probs)
        expected_loss = (
            1
            / 3
            * (
                math.log((1 / 3) / 0.1)
                + math.log((1 / 3) / 0.3)
                + math.log((1 / 3) / 0.6)
            )
        )
        assert loss == pytest.approx(expected_loss)

    def test_forward(self) -> None:
        torch.manual_seed(0)
        criterion = PMSNCustomLoss(target_distribution=_uniform_distribution)
        anchors = torch.rand((8 * 3, 10))
        targets = torch.rand((8, 10))
        prototypes = torch.rand((4, 10), requires_grad=True)
        criterion(anchors, targets, prototypes)

    @unittest.skipUnless(torch.cuda.is_available(), "cuda not available")
    def test_forward_cuda(self) -> None:
        torch.manual_seed(0)
        criterion = PMSNCustomLoss(target_distribution=_uniform_distribution)
        anchors = torch.rand((8 * 2, 10)).cuda()
        targets = torch.rand((8, 10)).cuda()
        prototypes = torch.rand((4, 10), requires_grad=True).cuda()
        criterion(anchors, targets, prototypes)


def test__power_law_distribution() -> None:
    power_dist = pmsn_loss._power_law_distribution(
        size=4, exponent=0.5, device=torch.device("cpu")
    )
    # 2.784457050376173 == sum(1/(k**0.5) for k in range(1, 5))
    assert torch.allclose(
        power_dist,
        torch.Tensor(
            [
                1 / (1**0.5),
                1 / (2**0.5),
                1 / (3**0.5),
                1 / (4**0.5),
            ]
        )
        / 2.784457050376173,
    )
    assert power_dist.device == torch.device("cpu")
    assert torch.allclose(power_dist.sum(), torch.Tensor([1.0]))


def _uniform_distribution(mean_anchor_probabilities: Tensor) -> Tensor:
    dim = mean_anchor_probabilities.shape[0]
    return mean_anchor_probabilities.new_ones(dim) / dim
