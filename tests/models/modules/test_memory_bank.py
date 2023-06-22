import re
import unittest

import pytest
import torch

from lightly.models.modules.memory_bank import MemoryBankModule


class TestNTXentLoss(unittest.TestCase):
    def test_init__negative_size(self):
        with self.assertRaises(ValueError):
            MemoryBankModule(size=-1)

    def test_forward_easy(self):
        bsz = 3
        dim, size = 2, 9
        n = 33 * bsz
        memory_bank = MemoryBankModule(size=size)

        ptr = 0
        for i in range(0, n, bsz):
            output = torch.randn(2 * bsz, dim)
            output.requires_grad = True
            out0, out1 = output[:bsz], output[bsz:]

            _, curr_memory_bank = memory_bank(out1, update=True)
            next_memory_bank = memory_bank.bank.transpose(0, -1)

            curr_diff = out0.T - curr_memory_bank[:, ptr : ptr + bsz]
            next_diff = out1.T - next_memory_bank[:, ptr : ptr + bsz]

            # the current memory bank should not hold the batch yet
            self.assertGreater(curr_diff.norm(), 1e-5)
            # the "next" memory bank should hold the batch
            self.assertGreater(1e-5, next_diff.norm())

            ptr = (ptr + bsz) % size

    def test_forward(self):
        bsz = 3
        dim, size = 2, 10
        n = 33 * bsz
        memory_bank = MemoryBankModule(size=size)

        for i in range(0, n, bsz):
            # see if there are any problems when the bank size
            # is no multiple of the batch size
            output = torch.randn(bsz, dim)
            _, _ = memory_bank(output)

    @unittest.skipUnless(torch.cuda.is_available(), "cuda not available")
    def test_forward__cuda(self):
        bsz = 3
        dim, size = 2, 10
        n = 33 * bsz
        memory_bank = MemoryBankModule(size=size)
        device = torch.device("cuda")
        memory_bank.to(device=device)

        for i in range(0, n, bsz):
            # see if there are any problems when the bank size
            # is no multiple of the batch size
            output = torch.randn(bsz, dim, device=device)
            _, _ = memory_bank(output)


class TestMemoryBank:
    def test_init__negative_size(self) -> None:
        with pytest.raises(
            ValueError,
            match="Illegal memory bank size -1, all entries must be non-negative.",
        ):
            MemoryBankModule(size=-1)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Illegal memory bank size (10, -1), all entries must be non-negative."
            ),
        ):
            MemoryBankModule(size=(10, -1))

    def test_init__no_dim_warning(self) -> None:
        with pytest.warns(
            UserWarning,
            match=re.escape(
                "Memory bank size 'size=10' does not specify feature "
                "dimension. It is recommended to set the feature dimension with "
                "'size=(n, dim)' when creating the memory bank. Distributed "
                "training might fail if the feature dimension is not set."
            ),
        ):
            MemoryBankModule(size=10)

    def test_forward(self) -> None:
        torch.manual_seed(0)
        memory_bank = MemoryBankModule(size=(5, 2), dim_first=False)
        x0 = torch.randn(3, 2)
        out0, bank0 = memory_bank(x0, update=True)
        # Verify that output is same as input.
        assert out0.tolist() == x0.tolist()
        # Verify that memory bank was initialized and has correct shape.
        assert bank0.shape == (5, 2)
        assert memory_bank.bank.shape == (5, 2)
        # Verify that output bank does not contain features from x0.
        assert bank0[:3].tolist() != x0.tolist()
        # Verify that memory bank was updated.
        assert memory_bank.bank[:3].tolist() == x0.tolist()

        x1 = torch.randn(3, 2)
        out1, bank1 = memory_bank(x1, update=True)
        # Verify that output is same as input.
        assert out1.tolist() == x1.tolist()
        # Verify that output bank contains features from x0.
        assert bank1[:3].tolist() == x0.tolist()
        # Verify that output bank does not contain features from x1.
        assert bank1[3:].tolist() != x1[:2].tolist()
        # Verify that memory bank was updated.
        assert memory_bank.bank[:3].tolist() == x0.tolist()
        assert memory_bank.bank[3:].tolist() == x1[:2].tolist()

        # At this point the memory bank is full.
        # Adding more features will start overwriting the bank from the beginning.

        x2 = torch.randn(3, 2)
        out2, bank2 = memory_bank(x2, update=True)
        # Verify that output is same as input.
        assert out2.tolist() == x2.tolist()
        # Verify that output bank contains features from x0 and x1.
        assert bank2[:3].tolist() == x0.tolist()
        assert bank2[3:].tolist() == x1[:2].tolist()
        # Verify that memory bank is overwritten.
        assert memory_bank.bank[:3].tolist() == x2.tolist()

    def test_forward__no_dim(self) -> None:
        torch.manual_seed(0)
        # Only specify size but not feature dimension.
        memory_bank = MemoryBankModule(size=5, dim_first=False)
        x0 = torch.randn(3, 2)
        out0, bank0 = memory_bank(x0, update=True)
        # Verify that output is same as input.
        assert out0.tolist() == x0.tolist()
        # Verify that memory bank was initialized and has correct shape.
        assert bank0.shape == (5, 2)
        assert memory_bank.bank.shape == (5, 2)
        # Verify that output bank does not contain features from x0.
        assert bank0[:3].tolist() != x0.tolist()
        # Verify that memory bank was updated.
        assert memory_bank.bank[:3].tolist() == x0.tolist()
