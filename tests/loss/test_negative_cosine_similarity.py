import pytest
import torch

from lightly.loss import NegativeCosineSimilarity


class TestNegativeCosineSimilarity:
    @pytest.mark.parametrize("bsz", range(1, 20))
    def test_forward_pass(self, bsz: int) -> None:
        loss = NegativeCosineSimilarity()
        x0 = torch.randn((bsz, 32))
        x1 = torch.randn((bsz, 32))

        # symmetry
        l1 = loss(x0, x1)
        l2 = loss(x1, x0)
        assert l1 == pytest.approx(l2, abs=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    @pytest.mark.parametrize("bsz", range(1, 20))
    def test_forward_pass_cuda(self, bsz: int) -> None:
        loss = NegativeCosineSimilarity()
        x0 = torch.randn((bsz, 32)).cuda()
        x1 = torch.randn((bsz, 32)).cuda()

        # symmetry
        l1 = loss(x0, x1)
        l2 = loss(x1, x0)
        assert l1 == pytest.approx(l2, abs=1e-5)
