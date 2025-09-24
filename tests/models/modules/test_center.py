import pytest
import torch
from torch import Tensor

from lightly.models.modules.center import Center


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
