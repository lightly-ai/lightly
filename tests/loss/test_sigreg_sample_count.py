"""SIGReg distributed sample counts remain exact for low-precision embeddings."""

from __future__ import annotations

import pytest
import torch
from pytest_mock import MockerFixture
from torch import Tensor
from torch import distributed as dist
from torch.distributed import nn as dist_nn

from lightly.loss import SIGReg
from lightly.loss.lejepa_loss import lightly_dist


@pytest.mark.parametrize(
    "dtype,batch_size",
    [
        (torch.bfloat16, 129),
        (torch.bfloat16, 257),
        (torch.float16, 2049),
        (torch.float16, 4097),
        (torch.float16, 32769),
        (torch.float32, 129),
        (torch.float64, 129),
        (torch.bfloat16, 8),
        (torch.float16, 8),
    ],
)
def test_sigreg_distributed_sample_count(
    mocker: MockerFixture, dtype: torch.dtype, batch_size: int
) -> None:
    """Compare two identical simulated ranks with an explicit global batch."""
    world_size = 2
    projections = torch.ones(batch_size, 2, dtype=dtype)
    distributed_loss = SIGReg(num_vectors=2, knots=3, gather_distributed=True)
    reference_loss = SIGReg(num_vectors=2, knots=3)
    counts: list[int] = []
    original_compute = distributed_loss._compute_cf_error_at_each_frequency

    def compute_error(x_t: Tensor, num_samples: int) -> Tensor:
        counts.append(num_samples)
        return original_compute(x_t=x_t, num_samples=num_samples)

    def reduce_count(tensor: Tensor) -> None:
        tensor.mul_(world_size)

    def reduce_statistics(tensor: Tensor) -> Tensor:
        return tensor * world_size

    def unit_vectors(
        device: torch.device, dtype: torch.dtype, num_features: int
    ) -> Tensor:
        # Older supported PyTorch versions lack a bfloat16 eye kernel.
        return torch.eye(num_features, device=device).to(dtype=dtype)

    mocker.patch.object(lightly_dist, "world_size", return_value=world_size)
    mocker.patch.object(dist, "all_reduce", side_effect=reduce_count)
    mocker.patch.object(dist_nn, "all_reduce", side_effect=reduce_statistics)
    mocker.patch.object(distributed_loss, "_generate_unit_vectors", new=unit_vectors)
    mocker.patch.object(reference_loss, "_generate_unit_vectors", new=unit_vectors)
    mocker.patch.object(
        distributed_loss, "_compute_cf_error_at_each_frequency", new=compute_error
    )

    actual = distributed_loss(projections)
    expected = reference_loss(projections.repeat(world_size, 1))

    assert counts == [world_size * batch_size]
    torch.testing.assert_close(actual, expected)
