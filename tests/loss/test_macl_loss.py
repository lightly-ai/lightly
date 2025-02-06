import pytest
import torch

from lightly.loss import MACLLoss

class TestMACLLoss:
    def test_forward_pass(self) -> None:
        loss = MACLLoss()
        for bsz in range(2, 20): # batch_size 1 is not allowed
            batch_1 = torch.randn((bsz, 32))
            batch_2 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            assert (l1 - l2).pow(2).item() == pytest.approx(0.0, abs=1e-10)

    def test_forward_pass_1d(self) -> None:
        loss = MACLLoss()
        for bsz in range(2, 20): # # batch_size 1 is not allowed
            batch_1 = torch.randn((bsz, 1))
            batch_2 = torch.randn((bsz, 1))

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            assert (l1 - l2).pow(2).item() == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_forward_pass_cuda(self) -> None:
        loss = MACLLoss()
        for bsz in range(2, 20): # batch_size 1 is not allowed
            batch_1 = torch.randn((bsz, 32)).cuda()
            batch_2 = torch.randn((bsz, 32)).cuda()

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            assert (l1 - l2).pow(2).item() == pytest.approx(0.0, abs=1e-10)