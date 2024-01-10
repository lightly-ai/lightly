import unittest

import pytest
import torch
from pytest_mock import MockerFixture
from torch import distributed as dist

from lightly.loss import SSLEYLoss


class TestSSLEYLoss:
    def test__init_gather_distributed(self, mocker: MockerFixture) -> None:
        mock_is_available = mocker.patch.object(dist, "is_available", return_value=True)
        SSLEYLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__init_gather_distributed_dist_not_available(
        self, mocker: MockerFixture
    ) -> None:
        mock_is_available = mocker.patch.object(
            dist, "is_available", return_value=False
        )
        with pytest.raises(ValueError):
            SSLEYLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test_forward(self) -> None:
        loss = SSLEYLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 32))
            x1 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss(x0, x1)
            l2 = loss(x1, x0)
            assert (l1 - l2).pow(2).item() == pytest.approx(0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available")
    def test_forward__cuda(self) -> None:
        loss = SSLEYLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 32)).cuda()
            x1 = torch.randn((bsz, 32)).cuda()

            # symmetry
            l1 = loss(x0, x1)
            l2 = loss(x1, x0)
            assert (l1 - l2).pow(2).item() == pytest.approx(0.0)

    def test_forward__error_batch_size_1(self) -> None:
        loss = SSLEYLoss()
        x0 = torch.randn((1, 32))
        x1 = torch.randn((2, 32))
        with pytest.raises(ValueError):
            loss(x0, x1)
        with pytest.raises(ValueError):
            loss(x1, x0)

    def test_forward__error_different_shapes(self) -> None:
        loss = SSLEYLoss()
        x0 = torch.randn((2, 32))
        x1 = torch.randn((2, 16))
        with pytest.raises(ValueError):
            loss(x0, x1)
