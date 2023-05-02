import unittest

import pytest
import torch
from pytest_mock import MockerFixture
from torch import distributed as dist

from lightly.loss.tico_loss import TiCoLoss


class TestTiCoLoss:
    def test__gather_distributed(self, mocker: MockerFixture) -> None:
        mock_is_available = mocker.patch.object(dist, "is_available", return_value=True)
        TiCoLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__gather_distributed_dist_not_available(
        self, mocker: MockerFixture
    ) -> None:
        mock_is_available = mocker.patch.object(
            dist, "is_available", return_value=False
        )
        with pytest.raises(ValueError):
            TiCoLoss(gather_distributed=True)
        mock_is_available.assert_called_once()


class TestTiCoLossUnitTest(unittest.TestCase):
    # Old tests in unittest style, please add new tests to TestTiCoLoss using pytest.
    def test_forward_pass(self):
        torch.manual_seed(0)
        loss = TiCoLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 256))
            x1 = torch.randn((bsz, 256))

            # symmetry
            l1 = loss(x0, x1, update_covariance_matrix=False)
            l2 = loss(x1, x0, update_covariance_matrix=False)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available")
    def test_forward_pass_cuda(self):
        torch.manual_seed(0)
        loss = TiCoLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 256)).cuda()
            x1 = torch.randn((bsz, 256)).cuda()

            # symmetry
            l1 = loss(x0, x1, update_covariance_matrix=False)
            l2 = loss(x1, x0, update_covariance_matrix=False)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0, 2)

    def test_forward_pass__error_batch_size_1(self):
        torch.manual_seed(0)
        loss = TiCoLoss()
        x0 = torch.randn((1, 256))
        x1 = torch.randn((1, 256))
        with self.assertRaises(AssertionError):
            loss(x0, x1, update_covariance_matrix=False)

    def test_forward_pass__error_different_shapes(self):
        torch.manual_seed(0)
        loss = TiCoLoss()
        x0 = torch.randn((2, 32))
        x1 = torch.randn((2, 16))
        with self.assertRaises(AssertionError):
            loss(x0, x1, update_covariance_matrix=False)
