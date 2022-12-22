import unittest
import copy

import torch
import torch.nn as nn

from lightly.models import utils
from lightly.models.utils import batch_shuffle
from lightly.models.utils import batch_unshuffle
from lightly.models.utils import activate_requires_grad
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.models.utils import normalize_weight
from lightly.models.utils import _no_grad_trunc_normal
from lightly.models.utils import cosine_decay_schedule


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

    def _assert_tensor_equal(self, x, y):
        # If the assertion fails then only an "assertion is not True" error is
        # shown without showing the contents of x and y. To help debugging, x
        # and y are printed. Note that the output is only shown if the assertion
        # fails.
        print(x)
        print(y)
        self.assertTrue(torch.equal(x, y))

    def test_batch_shuffle(self, seed=0):
        torch.manual_seed(seed)
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

    def test_repeat_token(self):
        token = torch.Tensor([[[1, 2, 3, 4]]])
        out = utils.repeat_token(token, size=(2, 3))
        self.assertEqual(tuple(out.shape), (2, 3, 4))
        self.assertListEqual(out[-1][-1].tolist(), [1, 2, 3, 4])

    def test_expand_index_like(self, seed=0):
        torch.manual_seed(seed)
        index = torch.Tensor([
            [1, 0, 3],
            [1, 2, 4],
        ]).long()
        tokens = torch.rand(2, 4, 5)
        expanded_index = utils.expand_index_like(index, tokens)

        self.assertEqual(tuple(expanded_index.shape), (2, 3, 5))

    def test_get_at_index(self, seed=0):
        torch.manual_seed(seed)
        index = torch.Tensor([
            [1, 0, 3],
            [1, 2, 0],
        ]).long()
        tokens = torch.rand(2, 4, 5)
        selected = utils.get_at_index(tokens, index)

        self.assertEqual(tuple(selected.shape), (2, 3, 5))

        # make sure that correct tokens were selected
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                self._assert_tensor_equal(tokens[i, index[i, j]], selected[i, j])

    def test_set_at_index(self, seed=0):
        torch.manual_seed(seed)
        index = torch.Tensor([
            [1, 0, 3],
            [1, 2, 0],
        ]).long()
        tokens = torch.rand(2, 4, 5)
        values = torch.rand(2, 3, 5)
        new_tokens = utils.set_at_index(tokens, index, values)

        # make sure that values are copied correctly
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                self._assert_tensor_equal(new_tokens[i, index[i, j]], values[i, j])

    def test_mask_at_index(self, seed=0):
        torch.manual_seed(seed)
        index = torch.Tensor([
            [1, 0, 3],
            [1, 2, 0],
        ]).long()
        tokens = torch.rand(2, 4, 5)
        mask_token = torch.rand(1, 1, 5)
        new_tokens = utils.mask_at_index(tokens.clone(), index.clone(), mask_token)
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                self._assert_tensor_equal(new_tokens[i, index[i, j]], mask_token[0, 0])


    def test_prepend_class_token(self, seed=0):
        torch.manual_seed(seed)
        tokens = torch.rand(2, 3, 5)
        class_token = torch.rand(1, 1, 5)
        new_tokens = utils.prepend_class_token(tokens, class_token)
        self.assertListEqual(list(new_tokens.shape), [2, 4, 5])

        # make sure that class token is inserted in correct place
        for i in range(new_tokens.shape[0]):
            self._assert_tensor_equal(new_tokens[i][0], class_token[0, 0])

    def test_patchify(self, seed=0):
        torch.manual_seed(seed)
        batch_size, channels, height, width = (2, 3, 8, 8)
        patch_size = 4
        images = torch.rand(batch_size, channels, height, width)
        batch_patches = utils.patchify(images, patch_size)

        height_patches = (height // patch_size)
        width_patches = (width // patch_size)
        num_patches = height_patches * width_patches
        patch_dim = channels * patch_size ** 2

        self.assertListEqual(list(batch_patches.shape), [batch_size, num_patches, patch_dim])

        # make sure that patches are correctly formed
        for (image, img_patches) in zip(images, batch_patches):
            for i in range(height_patches):
                for j in range(width_patches):
                    # extract patch from original image
                    expected_patch = image[:, i*patch_size : (i+1)*patch_size, j*patch_size : (j+1)*patch_size]
                    # permute and flatten to match order of patchified images
                    expected_patch = expected_patch.permute(1, 2, 0).flatten()
                    img_patch = img_patches[i * width_patches + j]
                    self._assert_tensor_equal(img_patch, expected_patch)

    def _test_random_token_mask(
        self, 
        seed=0, 
        mask_ratio=0.6, 
        mask_class_token=False, 
        device='cpu'
    ):
        torch.manual_seed(seed)
        batch_size, seq_length = 2, 5
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, seq_length),
            mask_ratio=mask_ratio,
            mask_class_token=mask_class_token,
            device=device,
        )

        # concatenating and sorting the two index tensors should result in a tensor
        # with every index appearing exactly once
        idx, _ = torch.cat([idx_keep, idx_mask], dim=1).sort(dim=1)
        expected_idx = torch.arange(seq_length).repeat(batch_size).reshape(batch_size, seq_length)
        expected_idx = expected_idx.to(device)
        self._assert_tensor_equal(idx, expected_idx)

        if not mask_class_token:
            # class token should be first in index
            self.assertTrue(torch.all(idx_keep[:, 0] == 0))

    def _test_random_token_mask_parameters(self, device):
        for mask_ratio in [0, 0.6, 1.0]:
            for mask_class_token in [False, True]:
                self._test_random_token_mask(
                    mask_ratio=mask_ratio,
                    mask_class_token=mask_class_token,
                    device=device,
                )

    def test_random_token_mask(self):
        self._test_random_token_mask_parameters(device='cpu')

    def test_cosine_decay_schedule(self):
        momentum_0 = cosine_decay_schedule(1, 10, 0.99, 1)
        momentum_hand_computed_0 = 0.99030154
        momentum_1 = cosine_decay_schedule(95, 100, 0.7, 2)
        momentum_hand_computed_1 = 1.99477063
        momentum_2 = cosine_decay_schedule(1, 1, 0.996, 1)
        momentum_hand_computed_2 = 1

        self.assertAlmostEqual(momentum_0, momentum_hand_computed_0, 6)
        self.assertAlmostEqual(momentum_1, momentum_hand_computed_1, 6)
        self.assertAlmostEqual(momentum_2, momentum_hand_computed_2, 6)

    @unittest.skipUnless(torch.cuda.is_available(), "No cuda available")
    def test_random_token_mask_cuda(self):
        self._test_random_token_mask_parameters(device="cuda")
