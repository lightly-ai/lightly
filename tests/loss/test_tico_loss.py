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

    def test_forward_pass(self) -> None:
        torch.manual_seed(0)
        loss = TiCoLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 256))
            x1 = torch.randn((bsz, 256))

            # symmetry
            l1 = loss(x0, x1, update_covariance_matrix=False)
            l2 = loss(x1, x0, update_covariance_matrix=False)
            assert torch.allclose(l1, l2, atol=1e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_forward_pass_cuda(self) -> None:
        torch.manual_seed(0)
        loss = TiCoLoss()
        for bsz in range(2, 4):
            x0 = torch.randn((bsz, 256)).cuda()
            x1 = torch.randn((bsz, 256)).cuda()

            # symmetry
            l1 = loss(x0, x1, update_covariance_matrix=False)
            l2 = loss(x1, x0, update_covariance_matrix=False)
            assert torch.allclose(l1, l2, atol=1e-2)

    def test_forward_pass__error_batch_size_1(self) -> None:
        torch.manual_seed(0)
        loss = TiCoLoss()
        x0 = torch.randn((1, 256))
        x1 = torch.randn((1, 256))
        with pytest.raises(AssertionError):
            loss(x0, x1, update_covariance_matrix=False)

    def test_forward_pass__error_different_shapes(self) -> None:
        torch.manual_seed(0)
        loss = TiCoLoss()
        x0 = torch.randn((2, 32))
        x1 = torch.randn((2, 16))
        with pytest.raises(AssertionError):
            loss(x0, x1, update_covariance_matrix=False)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("update_covariance_matrix", [False, True])
    def test_covariance_follows_dtype_conversion(
        self, dtype: torch.dtype, update_covariance_matrix: bool
    ) -> None:
        torch.manual_seed(0)
        initial_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        criterion = TiCoLoss()
        x0 = torch.randn((8, 16), dtype=initial_dtype)
        x1 = torch.randn((8, 16), dtype=initial_dtype)
        criterion(x0, x1)
        assert criterion.C is not None
        expected_covariance = criterion.C.to(dtype=dtype)

        criterion.to(dtype=dtype)
        assert criterion.C.dtype == dtype
        torch.testing.assert_close(criterion.C, expected_covariance)
        result = criterion(
            x0.to(dtype=dtype),
            x1.to(dtype=dtype),
            update_covariance_matrix=update_covariance_matrix,
        )
        reference = TiCoLoss()
        reference.C = expected_covariance.clone()
        expected = reference(
            x0.to(dtype=dtype),
            x1.to(dtype=dtype),
            update_covariance_matrix=update_covariance_matrix,
        )
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(criterion.C, reference.C)
        assert torch.isfinite(result)
        assert result.dtype == dtype
        assert criterion.C.dtype == dtype
        assert not criterion.C.requires_grad
        # Keep the rank-local EMA out of DDP broadcasts and old checkpoints.
        assert "C" not in dict(criterion.named_buffers())
        assert criterion.state_dict() == {}
        criterion.load_state_dict({})

    def test_dtype_conversion_before_initialization(self) -> None:
        criterion = TiCoLoss().double()
        assert "C" not in dict(criterion.named_buffers())
        x0 = torch.randn((8, 16), dtype=torch.float64)
        x1 = torch.randn((8, 16), dtype=torch.float64)
        assert torch.isfinite(criterion(x0, x1))
        assert criterion.C is not None
        assert criterion.C.dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_covariance_follows_device_conversion(self) -> None:
        criterion = TiCoLoss()
        x0 = torch.randn((8, 16))
        x1 = torch.randn((8, 16))
        criterion(x0, x1)
        assert criterion.C is not None
        expected_covariance = criterion.C.cuda()

        criterion.cuda()
        torch.testing.assert_close(criterion.C, expected_covariance)
        result = criterion(x0.cuda(), x1.cuda())
        assert result.is_cuda
        assert torch.isfinite(result)
