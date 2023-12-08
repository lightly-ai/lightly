import unittest

import pytest
import torch
from pytest_mock import MockerFixture
from torch import distributed as dist

from lightly.loss import SSLEYLoss


class TestSSLEYLoss:
    def test__gather_distributed(self, mocker: MockerFixture) -> None:
        mock_is_available = mocker.patch.object(dist, "is_available", return_value=True)
        SSLEYLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__gather_distributed_dist_not_available(
        self, mocker: MockerFixture
    ) -> None:
        mock_is_available = mocker.patch.object(
            dist, "is_available", return_value=False
        )
        with pytest.raises(ValueError):
            SSLEYLoss(gather_distributed=True)
        mock_is_available.assert_called_once()


class TestSSLEYLossUnitTest(unittest.TestCase):
    # Old tests in unittest style, please add new tests to TestSSLEYLoss using pytest.
    def test_forward_pass(self):
        loss = SSLEYLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 32))
            x1 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss(x0, x1)
            l2 = loss(x1, x0)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available")
    def test_forward_pass_cuda(self):
        loss = SSLEYLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 32)).cuda()
            x1 = torch.randn((bsz, 32)).cuda()

            # symmetry
            l1 = loss(x0, x1)
            l2 = loss(x1, x0)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0)

    def test_forward_pass__error_batch_size_1(self):
        loss = SSLEYLoss()
        x0 = torch.randn((1, 32))
        x1 = torch.randn((1, 32))
        with self.assertRaises(AssertionError):
            loss(x0, x1)

    def test_forward_pass__error_different_shapes(self):
        loss = SSLEYLoss()
        x0 = torch.randn((2, 32))
        x1 = torch.randn((2, 16))
        with self.assertRaises(AssertionError):
            loss(x0, x1)
