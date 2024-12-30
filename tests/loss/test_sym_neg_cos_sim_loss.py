import pytest
import torch

from lightly.loss import SymNegCosineSimilarityLoss


class TestSymNegCosineSimilarityLoss:
    @pytest.mark.parametrize("bsz", range(1, 20))
    def test_forward_pass(self, bsz: int) -> None:
        loss = SymNegCosineSimilarityLoss()
        z0 = torch.randn((bsz, 32))
        p0 = torch.randn((bsz, 32))
        z1 = torch.randn((bsz, 32))
        p1 = torch.randn((bsz, 32))

        # symmetry
        l1 = loss((z0, p0), (z1, p1))
        l2 = loss((z1, p1), (z0, p0))
        assert l1 == pytest.approx(l2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    @pytest.mark.parametrize("bsz", range(1, 20))
    def test_forward_pass_cuda(self, bsz: int) -> None:
        loss = SymNegCosineSimilarityLoss()
        z0 = torch.randn((bsz, 32)).cuda()
        p0 = torch.randn((bsz, 32)).cuda()
        z1 = torch.randn((bsz, 32)).cuda()
        p1 = torch.randn((bsz, 32)).cuda()

        # symmetry
        l1 = loss((z0, p0), (z1, p1))
        l2 = loss((z1, p1), (z0, p0))
        assert l1 == pytest.approx(l2)

    @pytest.mark.parametrize("bsz", range(1, 20))
    def test_neg_cosine_simililarity(self, bsz: int) -> None:
        loss = SymNegCosineSimilarityLoss()
        x = torch.randn((bsz, 32))
        y = torch.randn((bsz, 32))

        # symmetry
        l1 = loss._neg_cosine_simililarity(x, y)
        l2 = loss._neg_cosine_simililarity(y, x)
        assert l1 == pytest.approx(l2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    @pytest.mark.parametrize("bsz", range(1, 20))
    def test_neg_cosine_simililarity_cuda(self, bsz: int) -> None:
        loss = SymNegCosineSimilarityLoss()
        x = torch.randn((bsz, 32)).cuda()
        y = torch.randn((bsz, 32)).cuda()

        # symmetry
        l1 = loss._neg_cosine_simililarity(x, y)
        l2 = loss._neg_cosine_simililarity(y, x)
        assert l1 == pytest.approx(l2)
