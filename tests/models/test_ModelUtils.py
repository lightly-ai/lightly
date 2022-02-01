import unittest
import copy

import torch
import torch.nn as nn

from lightly.models.utils import batch_shuffle
from lightly.models.utils import batch_unshuffle
from lightly.models.utils import activate_requires_grad
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.models.utils import normalize_weight
from lightly.models.utils import _no_grad_trunc_normal


def has_grad(model: nn.Module):
    """Helper method to check if a model has `requires_grad` set to True
    """
    has_grad_ = False
    for param in model.parameters():
        if param.requires_grad == True:
            has_grad_ = True
            break
    return has_grad_


class TestModelUtils(unittest.TestCase):

    def test_batch_shuffle(self):
        x1 = torch.rand((4, 3,64,64))
        x1_shuffled, shuffle = batch_shuffle(x1)
        out1 = batch_unshuffle(x1_shuffled, shuffle)
        self.assertTrue(torch.equal(x1, out1))
        self.assertFalse(torch.equal(x1, x1_shuffled))

    def test_activate_requires_grad(self):
        model = nn.Sequential(
            nn.Linear(32,32),
            nn.ReLU(),
        )
        self.assertTrue(has_grad(model))
        deactivate_requires_grad(model)
        self.assertFalse(has_grad(model))
        activate_requires_grad(model)
        self.assertTrue(has_grad(model))
    
    def test_momentum_works(self):
        model = nn.Sequential(
            nn.Linear(32,32),
            nn.ReLU(),
        )
        model_momentum = copy.deepcopy(model)
        update_momentum(model, model_momentum, 0.99)

    def test_normalize_weight_linear(self):
        input_dim = 32
        output_dim = 64
        linear = nn.Linear(input_dim, output_dim, bias=False)
        normalize_weight(linear.weight, dim=0)
        self.assertEqual(linear.weight.norm(dim=0).sum(), input_dim)
        normalize_weight(linear.weight, dim=1)
        self.assertEqual(linear.weight.norm(dim=1).sum(), output_dim)

    def test_no_grad_trunc_normal(self, device="cpu", seed=0):
        torch.manual_seed(seed)
        tensor = torch.rand((8, 16)).to(device)
        a = -2
        b = 2
        _no_grad_trunc_normal(tensor, mean=0, std=1, a=-2, b=2)
        self.assertTrue(tensor.min() >= a)
        self.assertTrue(tensor.max() <= b)

    @unittest.skipUnless(torch.cuda.is_available(), "No cuda available")
    def test_no_grad_trunc_normal_cuda(self, seed=0):
        self.test_no_grad_trunc_normal(device="cuda")
