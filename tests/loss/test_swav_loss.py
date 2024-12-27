import pytest
import torch
from pytest_mock import MockerFixture
from torch import distributed as dist

from lightly.loss import SwaVLoss


class TestNTXentLoss:
    def test__sinkhorn_gather_distributed(self, mocker: MockerFixture) -> None:
        mock_is_available = mocker.patch.object(dist, "is_available", return_value=True)
        SwaVLoss(sinkhorn_gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__sinkhorn_gather_distributed_dist_not_available(
        self, mocker: MockerFixture
    ) -> None:
        mock_is_available = mocker.patch.object(
            dist, "is_available", return_value=False
        )
        with pytest.raises(ValueError):
            SwaVLoss(sinkhorn_gather_distributed=True)
        mock_is_available.assert_called_once()

    @pytest.mark.parametrize("n_low_res", range(6))
    @pytest.mark.parametrize("sinkhorn_iterations", range(3))
    def test_forward_pass(self, n_low_res: int, sinkhorn_iterations: int) -> None:
        n = 32
        n_high_res = 2
        high_res = [torch.eye(32, 32) for i in range(n_high_res)]
        criterion = SwaVLoss(sinkhorn_iterations=sinkhorn_iterations)
        low_res = [torch.eye(n, n) for i in range(n_low_res)]
        loss = criterion(high_res, low_res)
        # loss should be almost zero for unit matrix
        assert loss.cpu().numpy() < 0.5

    @pytest.mark.parametrize("n_low_res", range(6))
    @pytest.mark.parametrize("sinkhorn_iterations", range(3))
    def test_forward_pass_queue(self, n_low_res: int, sinkhorn_iterations: int) -> None:
        n = 32
        n_high_res = 2
        high_res = [torch.eye(32, 32) for i in range(n_high_res)]
        queue = [torch.eye(128, 32) for i in range(n_high_res)]
        criterion = SwaVLoss(sinkhorn_iterations=sinkhorn_iterations)
        low_res = [torch.eye(n, n) for i in range(n_low_res)]
        loss = criterion(high_res, low_res, queue)
        # loss should be almost zero for unit matrix
        assert loss.cpu().numpy() < 0.5

    @pytest.mark.parametrize("n_low_res", range(6))
    @pytest.mark.parametrize("sinkhorn_iterations", range(3))
    def test_forward_pass_bsz_1(self, n_low_res: int, sinkhorn_iterations: int) -> None:
        n = 32
        n_high_res = 2
        high_res = [torch.eye(1, n) for i in range(n_high_res)]
        criterion = SwaVLoss(sinkhorn_iterations=sinkhorn_iterations)
        low_res = [torch.eye(1, n) for i in range(n_low_res)]
        criterion(high_res, low_res)

    @pytest.mark.parametrize("n_low_res", range(6))
    @pytest.mark.parametrize("sinkhorn_iterations", range(3))
    def test_forward_pass_1d(self, n_low_res: int, sinkhorn_iterations: int) -> None:
        n = 32
        n_high_res = 2
        high_res = [torch.eye(n, 1) for i in range(n_high_res)]
        criterion = SwaVLoss(sinkhorn_iterations=sinkhorn_iterations)
        low_res = [torch.eye(n, 1) for i in range(n_low_res)]
        loss = criterion(high_res, low_res)
        # loss should be almost zero for unit matrix
        assert loss.cpu().numpy() < 0.5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    @pytest.mark.parametrize("n_low_res", range(6))
    @pytest.mark.parametrize("sinkhorn_iterations", range(3))
    def test_forward_pass_cuda(self, n_low_res: int, sinkhorn_iterations: int) -> None:
        n = 32
        n_high_res = 2
        high_res = [torch.eye(n, n).cuda() for i in range(n_high_res)]
        criterion = SwaVLoss(sinkhorn_iterations=sinkhorn_iterations)
        low_res = [torch.eye(n, n).cuda() for i in range(n_low_res)]
        loss = criterion(high_res, low_res)
        # loss should be almost zero for unit matrix
        assert loss.cpu().numpy() < 0.5
