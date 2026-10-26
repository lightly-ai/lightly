import pytest
import torch

from lightly.loss.hypersphere_loss import HypersphereLoss


class TestHyperSphereLoss:
    # NOTE: skipping bsz==1 case as its not relevant to this loss, and will produce nan-values
    @pytest.mark.parametrize("bsz", range(2, 20))
    def test_forward_pass(self, bsz: int) -> None:
        loss = HypersphereLoss()

        batch_1 = torch.randn((bsz, 32))
        batch_2 = torch.randn((bsz, 32))

        # symmetry
        l1 = loss(batch_1, batch_2)
        l2 = loss(batch_2, batch_1)

        assert l1 == pytest.approx(l2)
