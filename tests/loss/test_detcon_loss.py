from typing import Any, List, Tuple

import pytest
import torch
from torch import Tensor
from torch import distributed as dist

from lightly.loss import DetConBLoss, DetConSLoss


def get_detconb_example_input1() -> (
    Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
):
    pred1 = torch.tensor(
        [
            [[0.48962486, 0.2522575], [-0.6273904, -0.7026027]],
            [[0.7157335, -0.5684994], [-1.7461833, -2.7258704]],
        ]
    )

    pred2 = torch.tensor(
        [
            [[0.48962486, 0.2522575], [-0.6273904, -0.7026027]],
            [[0.7157335, -0.5684994], [-1.7461833, -2.7258704]],
        ]
    )

    target1 = torch.tensor(
        [
            [[0.48962486, 0.2522575], [-0.6273904, -0.7026027]],
            [[0.7157335, -0.5684994], [-1.7461833, -2.7258704]],
        ]
    )

    target2 = torch.tensor(
        [
            [[0.48962486, 0.2522575], [-0.6273904, -0.7026027]],
            [[0.7157335, -0.5684994], [-1.7461833, -2.7258704]],
        ]
    )

    mask1 = torch.tensor([[1, 0], [1, 1]])
    mask2 = torch.tensor([[1, 0], [1, 1]])
    return pred1, pred2, target1, target2, mask1, mask2


def get_detconb_example_output1() -> Tensor:
    return torch.tensor(3.012971)


def get_detconb_example_input2() -> (
    Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
):
    pred1 = torch.tensor(
        [
            [[-0.6067, -1.6712], [0.5318, -0.1990]],
            [[0.0800, -0.3512], [-0.3968, -1.0643]],
        ]
    )
    pred2 = torch.tensor(
        [
            [[-0.6067, -1.6712], [0.5318, -0.1990]],
            [[0.0800, -0.3512], [-0.3968, -1.0643]],
        ]
    )
    target1 = torch.tensor(
        [
            [[-0.6067, -1.6712], [0.5318, -0.1990]],
            [[0.0800, -0.3512], [-0.3968, -1.0643]],
        ]
    )
    target2 = torch.tensor(
        [
            [[-0.6067, -1.6712], [0.5318, -0.1990]],
            [[0.0800, -0.3512], [-0.3968, -1.0643]],
        ]
    )
    mask1 = torch.tensor([[0, 0], [0, 0]])
    mask2 = torch.tensor([[0, 0], [0, 0]])
    return pred1, pred2, target1, target2, mask1, mask2


def get_detconb_example_output2() -> Tensor:
    return torch.tensor(3.6086373)


def _fake_gather(tensor: Tensor, *args: Any, **kwargs: Any) -> List[Tensor]:
    if torch.isclose(tensor, get_detconb_example_input1()[2]).all():
        return [get_detconb_example_input1()[2], get_detconb_example_input2()[2]]
    elif torch.isclose(tensor, get_detconb_example_input1()[3]).all():
        return [get_detconb_example_input1()[3], get_detconb_example_input2()[3]]
    else:
        raise ValueError(
            f"unexpected input tensor: {tensor}, {get_detconb_example_input1()[2]}, {get_detconb_example_input1()[3]}"
        )


class TestDetConBLoss:
    def test_DetConBLoss_against_original_implementation(self) -> None:
        # test the loss against the original implementation. since it is in jax, we can only
        # test it for specific values here.
        # calculated with the original implementation

        loss = DetConBLoss(gather_distributed=False)
        loss_value = loss(*get_detconb_example_input1())
        assert torch.isclose(loss_value, get_detconb_example_output1(), atol=1e-4)

        loss_value = loss(*get_detconb_example_input2())
        assert torch.isclose(loss_value, get_detconb_example_output2(), atol=1e-4)

    def test_DetConBLoss_against_original_implementation_distributed(
        self, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: 0)
        monkeypatch.setattr(dist, "get_world_size", lambda: 2)
        monkeypatch.setattr(dist, "gather", _fake_gather)

        dist_loss = DetConBLoss(gather_distributed=True)
        dist_loss_value = dist_loss(*get_detconb_example_input1())
        assert dist_loss_value.shape == torch.Size([])
