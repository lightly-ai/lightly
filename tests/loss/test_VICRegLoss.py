import unittest

import pytest
import torch
import torch.nn.functional as F
from pytest_mock import MockerFixture
from torch import Tensor
from torch import distributed as dist

from lightly.loss import VICRegLoss


class TestVICRegLoss:
    def test__gather_distributed(self, mocker: MockerFixture) -> None:
        mock_is_available = mocker.patch.object(dist, "is_available", return_value=True)
        VICRegLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__gather_distributed_dist_not_available(
        self, mocker: MockerFixture
    ) -> None:
        mock_is_available = mocker.patch.object(
            dist, "is_available", return_value=False
        )
        with pytest.raises(ValueError):
            VICRegLoss(gather_distributed=True)
        mock_is_available.assert_called_once()


class TestVICRegLossUnitTest(unittest.TestCase):
    # Old tests in unittest style, please add new tests to TestVICRegLoss using pytest.
    def test_forward_pass(self):
        loss = VICRegLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 32))
            x1 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss(x0, x1)
            l2 = loss(x1, x0)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available")
    def test_forward_pass_cuda(self):
        loss = VICRegLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 32)).cuda()
            x1 = torch.randn((bsz, 32)).cuda()

            # symmetry
            l1 = loss(x0, x1)
            l2 = loss(x1, x0)
            self.assertAlmostEqual((l1 - l2).pow(2).item(), 0.0)

    def test_forward_pass__error_batch_size_1(self):
        loss = VICRegLoss()
        x0 = torch.randn((1, 32))
        x1 = torch.randn((1, 32))
        with self.assertRaises(AssertionError):
            loss(x0, x1)

    def test_forward_pass__error_different_shapes(self):
        loss = VICRegLoss()
        x0 = torch.randn((2, 32))
        x1 = torch.randn((2, 16))
        with self.assertRaises(AssertionError):
            loss(x0, x1)

    def test_forward__compare(self) -> None:
        # Compare against original implementation.
        loss = VICRegLoss()
        x0 = torch.randn((2, 32))
        x1 = torch.randn((2, 32))
        assert loss(x0, x1).item() == _reference_vicreg_loss(x0, x1).item()

    def test_forward__compare_vicregl(self) -> None:
        # Compare against implementation in VICRegL.
        # Note: nu_param is set to 0.5 because our loss implementation follows the
        # original VICReg implementation and there is a slight difference between the
        # implementations in VICReg and VICRegL.
        loss = VICRegLoss(nu_param=0.5)
        x0 = torch.randn((2, 10, 32))
        x1 = torch.randn((2, 10, 32))
        assert loss(x0, x1).item() == _reference_vicregl_vicreg_loss(x0, x1).item()


def _reference_vicreg_loss(
    x: Tensor,
    y: Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
):
    # Original VICReg loss from:
    # https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/main_vicreg.py#L194
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    batch_size = x.shape[0]
    num_features = x.shape[-1]
    repr_loss = F.mse_loss(x, y)

    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    cov_x = (x.T @ x) / (batch_size - 1)
    cov_y = (y.T @ y) / (batch_size - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features) + off_diagonal(
        cov_y
    ).pow_(2).sum().div(num_features)

    loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
    return loss


def _reference_vicregl_vicreg_loss(
    x: Tensor,
    y: Tensor,
    inv_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
) -> Tensor:
    # Loss implementation from VICRegL:
    # https://github.com/facebookresearch/VICRegL/blob/803ae4c8cd1649a820f03afb4793763e95317620/main_vicregl.py#L284
    repr_loss = inv_coeff * F.mse_loss(x, y)

    x = x - x.mean(0)
    y = y - y.mean(0)

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = var_coeff * (
        torch.mean(F.relu(1.0 - std_x)) / 2 + torch.mean(F.relu(1.0 - std_y)) / 2
    )

    x = x.permute((1, 0, 2))
    y = y.permute((1, 0, 2))

    *_, sample_size, num_channels = x.shape
    non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
    # Center features
    # centered.shape = NC
    x = x - x.mean(dim=-2, keepdim=True)
    y = y - y.mean(dim=-2, keepdim=True)

    cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
    cov_y = torch.einsum("...nc,...nd->...cd", y, y) / (sample_size - 1)
    cov_loss = (cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels) / 2 + (
        cov_y[..., non_diag_mask].pow(2).sum(-1) / num_channels
    ) / 2
    cov_loss = cov_loss.mean()
    cov_loss = cov_coeff * cov_loss

    return repr_loss + std_loss + cov_loss
