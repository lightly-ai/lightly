import unittest
import copy

import torch
import torch.nn as nn

from lightly.models.utils import batch_shuffle
from lightly.models.utils import batch_unshuffle
from lightly.models.utils import activate_requires_grad
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum

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
