import pytest
import torch

import lightly.models.modules.sparse_spark as sparse_spark
from torch import Tensor


def _create_mask() -> Tensor:
    return torch.tensor(
        [
            [
                [
                    [1, 0],
                    [0, 1],
                ]
            ]
        ],
        dtype=torch.bool,
    )


def test__get_active_ex_or_ii_expands_mask() -> None:
    H, W = 32, 32
    with sparse_spark.sparse_layer_context(_create_mask()):
        active = sparse_spark._get_active_ex_or_ii(H=H, W=W, returning_active_ex=True)
        assert not isinstance(active, tuple)

        assert active.shape == (1, 1, H, W)
        assert active[:, :, :16, :16].all()
        assert active[:, :, :16, 16:].logical_not().all()
        assert active[:, :, 16:, :16].logical_not().all()
        assert active[:, :, 16:, 16:].all()


def test__get_active_ex_or_ii_dont_shrink_mask() -> None:
    H, W = 4, 4
    with sparse_spark.sparse_layer_context(torch.ones(1, 1, 32, 32)):
        with pytest.raises(AssertionError):
            sparse_spark._get_active_ex_or_ii(H=H, W=W, returning_active_ex=False)


def test__get_active_ex_or_ii_raise_on_non_active_mask() -> None:
    H, W = 32, 32
    with pytest.raises(RuntimeError):
        sparse_spark._get_active_ex_or_ii(H=H, W=W, returning_active_ex=False)


def test__get_active_ex_or_ii_returning_ex_false_correct_values() -> None:
    H, W = 32, 32
    with sparse_spark.sparse_layer_context(_create_mask()):
        active_b, active_h, active_w = sparse_spark._get_active_ex_or_ii(
            H=H, W=W, returning_active_ex=False
        )
        active_ex = sparse_spark._get_active_ex_or_ii(
            H=H, W=W, returning_active_ex=True
        )
        assert not isinstance(active_ex, tuple)

        active_ex_scattered = torch.zeros_like(active_ex)
        active_ex_scattered[active_b, :, active_h, active_w] = 1

        assert torch.equal(active_ex, active_ex_scattered)


def test__sp_conv_forward() -> None:
    H, W = 32, 32
    with sparse_spark.sparse_layer_context(_create_mask()):
        conv = sparse_spark.SparseConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )
        conv.weight.data.fill_(1)
        assert conv.bias is not None
        conv.bias.data.fill_(0)

        x = torch.ones(1, 1, H, W)
        out = conv(x)

        assert out.shape == (1, 1, H, W)
        assert out[:, :, :16, :16].all()
        assert out[:, :, :16, 16:].logical_not().all()
        assert out[:, :, 16:, :16].logical_not().all()
        assert out[:, :, 16:, 16:].all()
