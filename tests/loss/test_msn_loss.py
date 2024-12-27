import pytest
import torch
import torch.nn.functional as F
from pytest_mock import MockerFixture
from torch import distributed as dist
from torch import nn
from torch.optim import SGD

from lightly.loss import msn_loss
from lightly.loss.msn_loss import MSNLoss
from lightly.models.modules.heads import MSNProjectionHead


class TestMSNLoss:
    def test__gather_distributed(self, mocker: MockerFixture) -> None:
        mock_is_available = mocker.patch.object(dist, "is_available", return_value=True)
        MSNLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__gather_distributed_dist_not_available(
        self, mocker: MockerFixture
    ) -> None:
        mock_is_available = mocker.patch.object(
            dist, "is_available", return_value=False
        )
        with pytest.raises(ValueError):
            MSNLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__init__temperature(self) -> None:
        MSNLoss(temperature=1.0)
        with pytest.raises(ValueError):
            MSNLoss(temperature=0.0)
        with pytest.raises(ValueError):
            MSNLoss(temperature=-1.0)

    def test__init__sinkhorn_iterations(self) -> None:
        MSNLoss(sinkhorn_iterations=0)
        with pytest.raises(ValueError):
            MSNLoss(sinkhorn_iterations=-1)

    def test__init__me_max_weight(self) -> None:
        criterion = MSNLoss(regularization_weight=0.0, me_max_weight=0.5)
        assert criterion.regularization_weight == 0.5

    def test_prototype_probabilitiy(self) -> None:
        torch.manual_seed(0)
        queries = F.normalize(torch.rand((8, 10)), dim=1)
        prototypes = F.normalize(torch.rand((4, 10)), dim=1)
        prob = msn_loss.prototype_probabilities(queries, prototypes, temperature=0.5)
        assert prob.shape == (8, 4)
        assert prob.max() == 1.0
        assert prob.min() > 0.0

        # verify sharpening
        prob1 = msn_loss.prototype_probabilities(queries, prototypes, temperature=0.1)
        # same prototypes should be assigned regardless of temperature
        assert torch.all(prob.argmax(dim=1) == prob1.argmax(dim=1))
        # probabilities of selected prototypes should be higher for lower temperature
        assert torch.all(prob.max(dim=1)[0] < prob1.max(dim=1)[0])

    def test_sharpen(self) -> None:
        torch.manual_seed(0)
        prob = torch.rand((8, 10))
        p0 = msn_loss.sharpen(prob, temperature=0.5)
        p1 = msn_loss.sharpen(prob, temperature=0.1)
        # indices of max probabilities should be the same regardless of temperature
        assert torch.all(p0.argmax(dim=1) == p1.argmax(dim=1))
        # max probabilities should be higher for lower temperature
        assert torch.all(p0.max(dim=1)[0] < p1.max(dim=1)[0])

    def test_sinkhorn(self) -> None:
        torch.manual_seed(0)
        prob = torch.rand((8, 10))
        out = msn_loss.sinkhorn(prob)
        assert torch.all(prob != out)

    def test_sinkhorn_no_iter(self) -> None:
        torch.manual_seed(0)
        prob = torch.rand((8, 10))
        out = msn_loss.sinkhorn(prob, iterations=0)
        assert torch.all(prob == out)

    @pytest.mark.parametrize("num_target_views", range(1, 4))
    def test_forward(self, num_target_views: int) -> None:
        torch.manual_seed(0)
        criterion = MSNLoss()
        anchors = torch.rand((8 * num_target_views, 10))
        targets = torch.rand((8, 10))
        prototypes = torch.rand((4, 10), requires_grad=True)
        criterion(anchors, targets, prototypes)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_forward_cuda(self) -> None:
        torch.manual_seed(0)
        criterion = MSNLoss()
        anchors = torch.rand((8 * 2, 10)).cuda()
        targets = torch.rand((8, 10)).cuda()
        prototypes = torch.rand((4, 10), requires_grad=True).cuda()
        criterion(anchors, targets, prototypes)

    def test_backward(self) -> None:
        torch.manual_seed(0)
        head = MSNProjectionHead(5, 16, 6)
        criterion = MSNLoss()
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
        assert torch.any(weights_before != weights_after)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_backward_cuda(self) -> None:
        torch.manual_seed(0)
        head = MSNProjectionHead(5, 16, 6)
        head.to("cuda")
        criterion = MSNLoss()
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
        assert torch.any(weights_before != weights_after)
