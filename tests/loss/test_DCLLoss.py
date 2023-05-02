import unittest
from unittest.mock import patch

import pytest
import torch
from pytest_mock import MockerFixture
from torch import distributed as dist

from lightly.loss.dcl_loss import DCLLoss, DCLWLoss, negative_mises_fisher_weights


class TestDCLLoss:
    def test__gather_distributed(self, mocker: MockerFixture) -> None:
        mock_is_available = mocker.patch.object(dist, "is_available", return_value=True)
        DCLLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__gather_distributed_dist_not_available(
        self, mocker: MockerFixture
    ) -> None:
        mock_is_available = mocker.patch.object(
            dist, "is_available", return_value=False
        )
        with pytest.raises(ValueError):
            DCLLoss(gather_distributed=True)
        mock_is_available.assert_called_once()


class TestDCLUnitTest(unittest.TestCase):
    # Old tests in unittest style, please add new tests to TestDCLLoss using pytest.
    def test_negative_mises_fisher_weights(self, seed=0):
        torch.manual_seed(seed)
        out0 = torch.rand((3, 5))
        out1 = torch.rand((3, 5))
        for sigma in [0.0000001, 0.5, 10000]:
            with self.subTest(sigma=sigma):
                negative_mises_fisher_weights(out0, out1, sigma)

    def test_dclloss_forward(self, seed=0):
        torch.manual_seed(seed=seed)
        for batch_size in [2, 3]:
            for dim in [1, 3]:
                out0 = torch.rand((batch_size, dim))
                out1 = torch.rand((batch_size, dim))
                for temperature in [0.1, 0.5, 1.0]:
                    for gather_distributed in [False, True]:
                        for weight_fn in [None, negative_mises_fisher_weights]:
                            with self.subTest(
                                batch_size=batch_size,
                                dim=dim,
                                temperature=temperature,
                                gather_distributed=gather_distributed,
                                weight_fn=weight_fn,
                            ):
                                criterion = DCLLoss(
                                    temperature=temperature,
                                    gather_distributed=gather_distributed,
                                    weight_fn=weight_fn,
                                )
                                loss0 = criterion(out0, out1)
                                loss1 = criterion(out1, out0)
                                self.assertGreater(loss0, 0)
                                self.assertAlmostEqual(loss0, loss1)

    def test_dclloss_backprop(self, seed=0):
        torch.manual_seed(seed=seed)
        out0 = torch.rand(3, 5)
        out1 = torch.rand(3, 5)
        layer = torch.nn.Linear(5, 5)
        out0 = layer(out0)
        out1 = layer(out1)
        criterion = DCLLoss()
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        loss = criterion(out0, out1)
        loss.backward()
        optimizer.step()

    def test_dclwloss_forward(self, seed=0):
        torch.manual_seed(seed=seed)
        out0 = torch.rand(3, 5)
        out1 = torch.rand(3, 5)
        criterion = DCLWLoss()
        loss0 = criterion(out0, out1)
        loss1 = criterion(out1, out0)
        self.assertGreater(loss0, 0)
        self.assertAlmostEqual(loss0, loss1)
