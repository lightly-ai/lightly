from torch import Tensor

from lightly.loss import PMSNCustomLoss, PMSNLoss


class TestPMSNLoss:
    def test__init__(self) -> None:
        criterion = PMSNLoss(power_law_exponent=0.5)
        assert criterion.target_distribution == "power_law"
        assert criterion.power_law_exponent == 0.5


class TestPMSNCustomLoss:
    def test__init__(self) -> None:
        def uniform_distribution(mean_anchor_probabilities: Tensor) -> Tensor:
            dim = mean_anchor_probabilities.shape[0]
            return mean_anchor_probabilities.new_ones(dim) / dim

        criterion = PMSNCustomLoss(target_distribution=uniform_distribution)
        assert criterion.target_distribution == uniform_distribution
