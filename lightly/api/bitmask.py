""" Module to work with Lightly BitMasks """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import List


def _hex_to_int(hexstring: str) -> int:
    """Converts a hex string representation of an integer to an integer.
    """
    return int(hexstring, 16)


def _bin_to_int(binstring: str) -> int:
    """Converts a binary string representation of an integer to an integer.
    """
    return int(binstring, 2)


def _int_to_hex(x: int) -> str:
    """Converts an integer to a hex string representation.
    """
    return hex(x)


def _int_to_bin(x: int) -> str:
    """Converts an integer to a binary string representation.
    """
    return bin(x)


def _get_nonzero_bits(x: int) -> List[int]:
    """Returns a list of indices of nonzero bits in x.
    """
    offset = 0
    nonzero_bit_indices = []
    while x > 0:
        # if the number is odd, there is a nonzero bit at offset
        if x % 2 > 0:
            nonzero_bit_indices.append(offset)
        # increment the offset and divide the number x by two (rounding down)
        offset += 1
        x = x // 2
    return nonzero_bit_indices


def _invert(x: int) -> int:
    """Flips every bit of x as if x was an unsigned integer.
    """
    # use XOR of x and 0xFFFFFF to get the inverse
    # return x ^ (2 ** (x.bit_length()) - 1)
    # TODO: the solution above can give wrong answers for the case where
    # the tag representation starts with a zero, therefore it needs to know
    # the exact number of samples in the dataset to do a correct inverse
    raise NotImplementedError('This method is not implemented yet...')


def _union(x: int, y: int) -> int:
    """Uses bitwise OR to get the union of the two masks.
    """
    return x | y


def _intersection(x: int, y: int) -> int:
    """Uses bitwise AND to get the intersection of the two masks.
    """
    return x & y


def _get_kth_bit(x: int, k: int) -> int:
    """Returns the kth bit in the mask from the right.
    """
    mask = 1 << k
    return x & mask


def _set_kth_bit(x: int, k: int) -> int:
    """Sets the kth bit in the mask from the right.
    """
    mask = 1 << k
    return x | mask


def _unset_kth_bit(x: int, k: int) -> int:
    """Clears the kth bit in the mask from the right.
    """
    mask = ~(1 << k)
    return x & mask


class BitMask:
    """Utility class to represent and manipulate tags.
    Attributes:
        x:
            An integer representation of the binary mask.
    Examples:
        >>> # the following are equivalent
        >>> mask = BitMask(6)
        >>> mask = BitMask.from_hex('0x6')
        >>> mask = Bitmask.from_bin('0b0110')
        >>> # for a dataset with 10 images, assume the following tag
        >>> # 0001011001 where the 1st, 4th, 5th and 7th image are selected
        >>> # this tag would be stored as 0x59.
        >>> hexstring = 0x59                    # what you receive from the api
        >>> mask = BitMask.from_hex(hexstring)  # create a bitmask from it
        >>> indices = mask.to_indices()         # get list of indices which are one
        >>> # indices is [0, 3, 4, 6]
    """

    def __init__(self, x):
        self.x = x

    @classmethod
    def from_hex(cls, hexstring: str):
        """Creates a bit mask object from a hexstring.
        """
        return cls(_hex_to_int(hexstring))

    @classmethod
    def from_bin(cls, binstring: str):
        """Creates a BitMask from a binary string.
        """
        return cls(_bin_to_int(binstring))

    def to_hex(self):
        """Creates a BitMask from a hex string.
        """
        return _int_to_hex(self.x)

    def to_bin(self):
        """Returns a binary string representing the bit mask.
        """
        return _int_to_bin(self.x)

    def to_indices(self) -> List[int]:
        """Returns the list of indices bits which are set to 1 from the right.
        Examples:
            >>> mask = BitMask('0b0101')
            >>> indices = mask.to_indices()
            >>> # indices is [0, 2]
        """
        return _get_nonzero_bits(self.x)

    def invert(self):
        """Sets every 0 to 1 and every 1 to 0 in the bitstring.

        """
        self.x = _invert(self.x)

    def complement(self):
        """Same as invert but with the appropriate name.
        """
        self.invert()

    def union(self, other):
        """Calculates the union of two bit masks.
        Examples:
            >>> mask1 = BitMask.from_bin('0b0011')
            >>> mask2 = BitMask.from_bin('0b1100')
            >>> mask1.union(mask2)
            >>> # mask1.binstring is '0b1111'
        """
        self.x = _union(self.x, other.x)

    def intersection(self, other):
        """Calculates the intersection of two bit masks.
        Examples:
            >>> mask1 = BitMask.from_bin('0b0011')
            >>> mask2 = BitMask.from_bin('0b1100')
            >>> mask1.intersection(mask2)
            >>> # mask1.binstring is '0b0000'
        """
        self.x = _intersection(self.x, other.x)

    def get_kth_bit(self, k: int) -> bool:
        """Returns the boolean value of the kth bit from the right.
        """
        return _get_kth_bit(self.x, k) > 0

    def set_kth_bit(self, k: int):
        """Sets the kth bit from the right to '1'.
        Examples:
            >>> mask = BitMask('0b0000')
            >>> mask.set_kth_bit(2)
            >>> # mask.binstring is '0b0100'
        """
        self.x = _set_kth_bit(self.x, k)

    def unset_kth_bit(self, k: int):
        """Unsets the kth bit from the right to '0'.
        Examples:
            >>> mask = BitMask('0b1111')
            >>> mask.unset_kth_bit(2)
            >>> # mask.binstring is '0b1011'
        """
        self.x = _unset_kth_bit(self.x, k)