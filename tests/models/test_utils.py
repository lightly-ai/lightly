import unittest
import torch
import torch.nn as nn

from lightly.models.utils import normalize_weight


class TestModelsUtils(unittest.TestCase):

    def test_normalize_weight_linear(self):

        input_dim = 32
        output_dim = 64

        linear = nn.Linear(input_dim, output_dim, bias=False)

        normalize_weight(linear.weight, dim=0)
        self.assertEqual(linear.weight.norm(dim=0).sum(), input_dim)

        normalize_weight(linear.weight, dim=1)
        self.assertEqual(linear.weight.norm(dim=1).sum(), output_dim)
