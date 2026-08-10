import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytest_mock import MockerFixture
from torch import Tensor
from torch import distributed as torch_dist

from lightly.loss.koleo_loss import KoLeoLoss


class TestKoLeoLoss:
    # Test values generated using the original implementation from:
    # https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py
    @pytest.mark.parametrize(
        "x, expected_loss",
        [
            (torch.tensor([[1.0]]), 17.7275),
            (torch.tensor([[0.0, 1.0], [0.0, -1.0]]), -math.log(2)),
            (torch.tensor([[0.0, 1.0], [1.0, 0.0]]), -math.log(2**0.5)),
            (
                torch.tensor(
                    [
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [-1.0, 0.0],
                        [0.0, -1.0],
                        [-1.0, -1.0],
                    ]
                ),
                0.2674,
            ),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_forward(self, x: Tensor, expected_loss: float, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping test.")

        x = x.to(device)
        loss = KoLeoLoss().to(device)
        assert loss(x).item() == pytest.approx(expected_loss, rel=1e-4)

    def test_forward__group_size_none_is_full_batch(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(8, 4)
        assert KoLeoLoss(group_size=8)(x).item() == pytest.approx(KoLeoLoss()(x).item())

    def test_forward__group_size(self) -> None:
        """Groups are consecutive chunks of the batch and are averaged over."""
        torch.manual_seed(0)
        x = torch.randn(8, 4)

        loss = KoLeoLoss(group_size=4)(x)

        loss_fn = KoLeoLoss()
        expected = 0.5 * (loss_fn(x[:4]) + loss_fn(x[4:]))
        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_forward__group_size_finds_neighbors_within_group(self) -> None:
        """A neighbor in another group is ignored.

        The batch holds two copies of the same feature in different groups. Without
        grouping the nearest neighbor is the identical feature at distance zero, which
        gives a much larger loss than the within-group neighbor.
        """
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        assert KoLeoLoss(group_size=2)(x).item() == pytest.approx(
            -math.log(2**0.5), rel=1e-4
        )
        assert KoLeoLoss()(x).item() > 10.0

    @pytest.mark.parametrize("topk", [1, 2, 3])
    def test_forward__topk(self, topk: int) -> None:
        """The k nearest neighbors of every feature are penalized."""
        torch.manual_seed(0)
        x = F.normalize(torch.randn(6, 4), dim=-1)

        loss = KoLeoLoss(topk=topk)(x)

        cos_sim = x @ x.t()
        cos_sim.fill_diagonal_(-2)
        nn_idx = cos_sim.topk(k=topk, dim=-1).indices
        distances = torch.stack(
            [(x - x[idx]).norm(dim=-1) for idx in nn_idx.unbind(dim=-1)]
        )
        expected = -(distances + 1e-8).log().mean()
        assert loss.item() == pytest.approx(expected.item(), rel=1e-4)

    def test_forward__empty_batch(self) -> None:
        with pytest.raises(ValueError, match="non-empty batch"):
            KoLeoLoss()(torch.randn(0, 4))

    def test_forward__batch_size_not_divisible_by_group_size(self) -> None:
        with pytest.raises(ValueError, match="must be divisible by group size"):
            KoLeoLoss(group_size=3)(torch.randn(8, 4))

    @pytest.mark.parametrize("topk, group_size", [(4, 4), (5, 4), (2, 1)])
    def test_forward__topk_too_large_for_group(
        self, topk: int, group_size: int
    ) -> None:
        with pytest.raises(ValueError, match="must not be larger than"):
            KoLeoLoss(topk=topk, group_size=group_size)(torch.randn(8, 4))

    @pytest.mark.parametrize("topk, group_size", [(0, None), (1, 0)])
    def test__init__invalid_parameters(self, topk: int, group_size: int) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            KoLeoLoss(topk=topk, group_size=group_size)

    def test__init__gather_distributed_dist_not_available(
        self, mocker: MockerFixture
    ) -> None:
        mock_is_available = mocker.patch.object(
            torch_dist, "is_available", return_value=False
        )
        with pytest.raises(ValueError, match="torch.distributed is not available"):
            KoLeoLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test_gather_distributed_world_size_one_does_not_gather(
        self, mocker: MockerFixture
    ) -> None:
        mock_gather = mocker.patch("lightly.loss.koleo_loss.lightly_dist.gather")

        torch.manual_seed(0)
        loss = KoLeoLoss(gather_distributed=True)(torch.randn(4, 8))

        assert loss.isfinite()
        mock_gather.assert_not_called()

    def test_gather_distributed_matches_non_distributed(
        self, mocker: MockerFixture
    ) -> None:
        """Gathered forward equals non-distributed forward on the global batch.

        Simulates ``world_size=2`` with different data on both ranks, so the global
        batch is the concatenation of the two local batches.
        """
        torch.manual_seed(0)
        rank_0, rank_1 = torch.randn(4, 8), torch.randn(4, 8)

        # Non-distributed truth: loss on the concatenated global batch.
        expected = KoLeoLoss(group_size=4)(torch.cat([rank_0, rank_1]))

        mocker.patch("lightly.loss.koleo_loss.lightly_dist.world_size", return_value=2)
        # gather is called with the already normalized features of the local rank.
        mocker.patch(
            "lightly.loss.koleo_loss.lightly_dist.gather",
            side_effect=lambda tensor: (
                tensor,
                F.normalize(rank_1, p=2, dim=-1, eps=1e-8),
            ),
        )
        loss = KoLeoLoss(group_size=4, gather_distributed=True)(rank_0)

        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_gather_distributed_gradient_matches_non_distributed(
        self, mocker: MockerFixture
    ) -> None:
        """The gradient w.r.t. the local features matches the non-distributed one.

        A gathered forward that is correct can still produce a wrong backward, see
        #1977. GatherLayer itself is mocked here, so this covers that grouping and
        neighbor indexing keep the gradients attached to the right features.
        """
        torch.manual_seed(0)
        rank_0 = torch.randn(4, 8)
        rank_1 = torch.randn(4, 8)

        # Non-distributed truth: gradient of the loss on the concatenated global batch.
        global_batch = torch.cat([rank_0, rank_1]).requires_grad_()
        KoLeoLoss(group_size=4)(global_batch).backward()
        assert global_batch.grad is not None
        expected_grad = global_batch.grad[: len(rank_0)]

        mocker.patch("lightly.loss.koleo_loss.lightly_dist.world_size", return_value=2)
        mocker.patch(
            "lightly.loss.koleo_loss.lightly_dist.gather",
            side_effect=lambda tensor: (
                tensor,
                F.normalize(rank_1, p=2, dim=-1, eps=1e-8),
            ),
        )
        local_batch = rank_0.clone().requires_grad_()
        KoLeoLoss(group_size=4, gather_distributed=True)(local_batch).backward()

        assert local_batch.grad is not None
        assert torch.allclose(local_batch.grad, expected_grad, atol=1e-6)
