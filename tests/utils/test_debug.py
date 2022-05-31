import unittest
import torch
import math

from lightly.utils import debug

BATCH_SIZE = 10
DIMENSION = 10

class TestDist(unittest.TestCase):

    def test_std_of_l2_normalized_collapsed(self):
        z = torch.ones(BATCH_SIZE, DIMENSION) # collapsed output
        self.assertEqual(debug.std_of_l2_normalized(z), 0.0)

    def test_std_of_l2_normalized_uniform(self, eps: float = 1e-5):
        z = torch.eye(BATCH_SIZE)
        self.assertLessEqual(
            abs(debug.std_of_l2_normalized(z) - 1 / math.sqrt(z.shape[1])),
            eps,
        )

    def test_std_of_l2_normalized_raises(self):
        z = torch.zeros(BATCH_SIZE)
        with self.assertRaises(ValueError):
            debug.std_of_l2_normalized(z)
        z = torch.zeros(BATCH_SIZE, BATCH_SIZE, DIMENSION)
        with self.assertRaises(ValueError):
            debug.std_of_l2_normalized(z)      