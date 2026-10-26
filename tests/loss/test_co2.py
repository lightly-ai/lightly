import pytest
import torch

from lightly.loss.regularizer import CO2Regularizer


class TestCO2Regularizer:
    @pytest.mark.parametrize("bsz", range(1, 20))
    def test_forward_pass_no_memory_bank(self, bsz: int) -> None:
        reg = CO2Regularizer(memory_bank_size=0)
        batch_1 = torch.randn((bsz, 32))
        batch_2 = torch.randn((bsz, 32))

        # symmetry
        l1 = reg(batch_1, batch_2)
        l2 = reg(batch_2, batch_1)
        assert torch.allclose(l1, l2)

    @pytest.mark.parametrize("bsz", range(1, 20))
    def test_forward_pass_memory_bank(self, bsz: int) -> None:
        reg = CO2Regularizer(memory_bank_size=(4096, 32))
        batch_1 = torch.randn((bsz, 32))
        batch_2 = torch.randn((bsz, 32))

        l1 = reg(batch_1, batch_2)
        assert l1 > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    @pytest.mark.parametrize("bsz", range(1, 20))
    def test_forward_pass_cuda_no_memory_bank(self, bsz: int) -> None:
        reg = CO2Regularizer(memory_bank_size=0)
        batch_1 = torch.randn((bsz, 32)).cuda()
        batch_2 = torch.randn((bsz, 32)).cuda()

        # symmetry
        l1 = reg(batch_1, batch_2)
        l2 = reg(batch_2, batch_1)
        assert torch.allclose(l1, l2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    @pytest.mark.parametrize("bsz", range(1, 20))
    def test_forward_pass_cuda_memory_bank(self, bsz: int) -> None:
        reg = CO2Regularizer(memory_bank_size=(4096, 32))
        batch_1 = torch.randn((bsz, 32)).cuda()
        batch_2 = torch.randn((bsz, 32)).cuda()

        # symmetry
        l1 = reg(batch_1, batch_2)
        assert l1 > 0
