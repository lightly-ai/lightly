import unittest
from lightly.api.bitmask import BitMask

N = 10


class TestBitMask(unittest.TestCase):

    def setup(self, psuccess=1.):
        pass

    def test_get_and_set(self):

        mask = BitMask.from_bin("0b11110000")

        self.assertFalse(mask.get_kth_bit(2))
        mask.set_kth_bit(2)
        self.assertTrue(mask.get_kth_bit(2))

        self.assertTrue(mask.get_kth_bit(4))
        mask.unset_kth_bit(4)
        self.assertFalse(mask.get_kth_bit(4))

    def test_large_bitmasks(self):
        bitstring = "0b" + "1" * 5678
        mask = BitMask.from_bin(bitstring)
        mask_as_bitstring = mask.to_bin()
        self.assertEqual(mask_as_bitstring, bitstring)

    def test_bitmask_from_length(self):
        length = 4
        mask = BitMask.from_length(length)
        self.assertEqual(mask.to_bin(), "0b1111")

    def test_get_and_set_outside_of_range(self):

        mask = BitMask.from_bin("0b11110000")

        self.assertFalse(mask.get_kth_bit(100))
        mask.set_kth_bit(100)
        self.assertTrue(mask.get_kth_bit(100))

    def test_inverse(self):
        # TODO: proper implementation
        return

        x = int("0b11110000", 2)
        y = int("0b00001111", 2)
        mask = BitMask(x)
        mask.invert()
        self.assertEqual(mask.x, y)

        x = int("0b010101010101010101", 2)
        y = int("0b101010101010101010", 2)
        mask = BitMask(x)
        mask.invert()
        self.assertEqual(mask.x, y)

    def test_store_and_retrieve(self):

        x = int("0b01010100100100100100100010010100100100101001001010101010", 2)
        mask = BitMask(x)
        mask.set_kth_bit(11)
        mask.set_kth_bit(22)
        mask.set_kth_bit(33)
        mask.set_kth_bit(44)
        mask.set_kth_bit(55)
        mask.set_kth_bit(66)
        mask.set_kth_bit(77)
        mask.set_kth_bit(88)
        mask.set_kth_bit(99)

        somewhere = mask.to_hex()
        somewhere_else = mask.to_bin()

        mask_somewhere = BitMask.from_hex(somewhere)
        mask_somewhere_else = BitMask.from_bin(somewhere_else)

        self.assertEqual(mask.x, mask_somewhere.x)
        self.assertEqual(mask.x, mask_somewhere_else.x)

    def test_union(self):
        mask_a = BitMask.from_bin("0b001")
        mask_b = BitMask.from_bin("0b100")
        mask_a.union(mask_b)
        self.assertEqual(mask_a.x, int("0b101", 2))

    def test_intersection(self):
        mask_a = BitMask.from_bin("0b101")
        mask_b = BitMask.from_bin("0b100")
        mask_a.intersection(mask_b)
        self.assertEqual(mask_a.x, int("0b100", 2))

    def test_difference(self):
        mask_a = BitMask.from_bin("0b101")
        mask_b = BitMask.from_bin("0b001")
        mask_a.difference(mask_b)
        self.assertEqual(mask_a.x, int("0b100", 2))

    def test_subset_a_list(self):
        list_ = [4, 7, 9, 1]
        mask = BitMask.from_bin("0b0101")
        target_masked_list = [7, 1]
        masked_list = mask.subset_a_list(list_)
        self.assertEqual(target_masked_list,masked_list)


    def test_nonzero_bits(self):

        mask = BitMask.from_bin("0b0")
        indices = [100, 1000, 10_000, 100_000]

        self.assertEqual(mask.x, 0)
        for index in indices:
            mask.set_kth_bit(index)

        self.assertGreaterEqual(mask.x, 0)
        also_indices = mask.to_indices()

        for i, j in zip(indices, also_indices):
            self.assertEqual(i, j)
