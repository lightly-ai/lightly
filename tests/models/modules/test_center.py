from __future__ import annotations

import typing

import pytest
import torch
from torch import Tensor
from torch import distributed as dist
from torch.nn import functional as F

from lightly.models.modules.center import Center, sinkhorn_knopp
from tests.ddp_helpers import NUM_PROCESSES, USE_PYTEST_POOL


@typing.no_type_check
@torch.no_grad()
def original_sinkhorn_knopp_teacher(teacher_output, teacher_temp, n_iterations=3):
    """Copy paste from the original DINOv3 implementation. We use this to verify our
    implementation.

    The only change from the original code is that distributed training is no longer
    assumed.

    Source: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/loss/dino_clstoken_loss.py#L43
    """
    teacher_output = teacher_output.float()
    world_size = 1
    Q = torch.exp(teacher_output / teacher_temp).t()
    B = Q.shape[1] * world_size
    K = Q.shape[0]

    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for _ in range(n_iterations):
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B
    return Q.t()


class TestCenter:
    def test__init__invalid_mode(self) -> None:
        with pytest.raises(ValueError):
            Center(size=(1, 32), mode="invalid")

    def test_value(self) -> None:
        center = Center(size=(1, 32), mode="mean")
        assert torch.all(center.value == 0)

    @pytest.mark.parametrize(
        "x, expected",
        [
            (torch.tensor([[0.0, 0.0], [0.0, 0.0]]), torch.tensor([0.0, 0.0])),
            (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([2.0, 3.0])),
        ],
    )
    def test_update(self, x: Tensor, expected: Tensor) -> None:
        center = Center(size=(1, 2), mode="mean", momentum=0.0)
        center.update(x)
        assert torch.all(center.value == expected)

    @pytest.mark.parametrize(
        "momentum, expected",
        [
            (0.0, torch.tensor([1.0, 2.0])),
            (0.1, torch.tensor([0.9, 1.8])),
            (0.5, torch.tensor([0.5, 1.0])),
            (1.0, torch.tensor([0.0, 0.0])),
        ],
    )
    def test_update__momentum(self, momentum: float, expected: Tensor) -> None:
        center = Center(size=(1, 2), mode="mean", momentum=momentum)
        center.update(torch.tensor([[1.0, 2.0]]))
        assert torch.all(center.value == expected)


def _teacher_logits(batch_size: int, num_prototypes: int, dim: int = 16) -> Tensor:
    """Returns logits as produced by a DINO projection head.

    The head L2-normalizes the features before the weight-normed prototype layer, so
    the logits are cosine similarities and lie in [-1, 1].
    """
    features = F.normalize(torch.randn(batch_size, dim), dim=-1)
    prototypes = F.normalize(torch.randn(dim, num_prototypes), dim=0)
    return features @ prototypes


def _sinkhorn_knopp_worker(x: Tensor, split_sizes: list[int]) -> tuple[int, Tensor]:
    # Pool worker: Sinkhorn-Knopp on the part of x that belongs to this rank, returns
    # the rank and the probabilities. See #1982. The rank is read from the process
    # group because the pool hands every task to whichever worker is free.
    rank = dist.get_rank()
    return rank, sinkhorn_knopp(x=x.split(split_sizes)[rank], temperature=0.04)


class TestSinkhornKnopp:
    @pytest.mark.parametrize("temperature", [0.04, 0.07, 0.1])
    @pytest.mark.parametrize("num_iterations", [1, 3, 5])
    def test__matches_original_implementation(
        self, temperature: float, num_iterations: int
    ) -> None:
        torch.manual_seed(0)
        x = _teacher_logits(batch_size=16, num_prototypes=8)

        expected = original_sinkhorn_knopp_teacher(
            teacher_output=x, teacher_temp=temperature, n_iterations=num_iterations
        )
        probabilities = sinkhorn_knopp(
            x=x, temperature=temperature, num_iterations=num_iterations
        )
        assert torch.allclose(probabilities, expected, atol=1e-6)

    def test__rows_sum_to_one(self) -> None:
        torch.manual_seed(0)
        x = _teacher_logits(batch_size=16, num_prototypes=8)
        probabilities = sinkhorn_knopp(x=x, temperature=0.04)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(16), atol=1e-5)

    def test__prototypes_are_equally_used(self) -> None:
        """Every prototype gets the same total weight across the batch."""
        torch.manual_seed(0)
        batch_size, num_prototypes = 32, 8
        probabilities = sinkhorn_knopp(
            x=_teacher_logits(batch_size=batch_size, num_prototypes=num_prototypes),
            temperature=0.1,
            num_iterations=100,
        )
        expected = torch.full((num_prototypes,), batch_size / num_prototypes)
        assert torch.allclose(probabilities.sum(dim=0), expected, atol=1e-2)

    def test__no_grad(self) -> None:
        x = _teacher_logits(batch_size=4, num_prototypes=8).requires_grad_()
        assert not sinkhorn_knopp(x=x, temperature=0.04).requires_grad

    @pytest.mark.DDP
    @pytest.mark.skipif(not USE_PYTEST_POOL, reason="DDP pool is not available")
    @pytest.mark.parametrize("equal_batch_sizes", [True, False])
    def test__gather_distributed_matches_non_distributed(
        self, equal_batch_sizes: bool
    ) -> None:
        # The normalization over all processes equals the non-distributed one on the
        # concatenated batch. This must also hold if the processes have different
        # batch sizes, as with the masked tokens of the iBOT loss.
        torch.manual_seed(0)
        split_sizes = [
            8 if equal_batch_sizes else 4 * (rank + 1) for rank in range(NUM_PROCESSES)
        ]
        x = _teacher_logits(batch_size=sum(split_sizes), num_prototypes=8)

        results = dict(
            pytest.pool.starmap(  # type: ignore[attr-defined]
                _sinkhorn_knopp_worker, [(x, split_sizes)] * NUM_PROCESSES
            )
        )
        expected = sinkhorn_knopp(x=x, temperature=0.04, gather_distributed=False)

        assert sorted(results) == list(range(NUM_PROCESSES))
        probabilities = torch.cat([results[rank] for rank in range(NUM_PROCESSES)])
        assert torch.allclose(probabilities, expected, atol=1e-6)
