import pytest
from pytest_mock import MockerFixture
from torch import distributed as dist

from lightly.loss.barlow_twins_loss import BarlowTwinsLoss


class TestBarlowTwinsLoss:
    def test__gather_distributed(self, mocker: MockerFixture) -> None:
        mock_is_available = mocker.patch.object(dist, "is_available", return_value=True)
        BarlowTwinsLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__gather_distributed_dist_not_available(
        self, mocker: MockerFixture
    ) -> None:
        mock_is_available = mocker.patch.object(
            dist, "is_available", return_value=False
        )
        with pytest.raises(ValueError):
            BarlowTwinsLoss(gather_distributed=True)
        mock_is_available.assert_called_once()
