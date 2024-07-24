import copy
import random
import unittest
from typing import List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
from torch.nn import Identity

from lightly.models import utils
from lightly.models.utils import (
    _no_grad_trunc_normal,
    activate_requires_grad,
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    nearest_neighbors,
    normalize_weight,
    update_momentum,
)


def has_grad(model: nn.Module):
    """Helper method to check if a model has `requires_grad` set to True"""
    has_grad_ = False
    for param in model.parameters():
        if param.requires_grad == True:
            has_grad_ = True
            break
    return has_grad_


class TestModelUtils(unittest.TestCase):
    def _assert_tensor_equal(self, x, y):
        # If the assertion fails then only an "assertion is not True" error is
        # shown without showing the contents of x and y. To help debugging, x
        # and y are printed. Note that the output is only shown if the assertion
        # fails.
        print(x)
        print(y)
        self.assertTrue(torch.equal(x, y))

    def test_batch_shuffle(self, seed=0):
        torch.manual_seed(seed)
        x1 = torch.rand((4, 3, 64, 64))
        x1_shuffled, shuffle = batch_shuffle(x1)
        out1 = batch_unshuffle(x1_shuffled, shuffle)
        self.assertTrue(torch.equal(x1, out1))
        self.assertFalse(torch.equal(x1, x1_shuffled))

    def test_activate_requires_grad(self):
        model = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.assertTrue(has_grad(model))
        deactivate_requires_grad(model)
        self.assertFalse(has_grad(model))
        activate_requires_grad(model)
        self.assertTrue(has_grad(model))

    def test_momentum_works(self):
        model = nn.Sequential(
            nn.Linear(32, 32),
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
        index = torch.Tensor(
            [
                [1, 0, 3],
                [1, 2, 4],
            ]
        ).long()
        tokens = torch.rand(2, 4, 5)
        expanded_index = utils.expand_index_like(index, tokens)

        self.assertEqual(tuple(expanded_index.shape), (2, 3, 5))

    def test_get_at_index(self, seed=0):
        torch.manual_seed(seed)
        index = torch.Tensor(
            [
                [1, 0, 3],
                [1, 2, 0],
            ]
        ).long()
        tokens = torch.rand(2, 4, 5)
        selected = utils.get_at_index(tokens, index)

        self.assertEqual(tuple(selected.shape), (2, 3, 5))

        # make sure that correct tokens were selected
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                self._assert_tensor_equal(tokens[i, index[i, j]], selected[i, j])

    def test_set_at_index(self, seed=0):
        torch.manual_seed(seed)
        index = torch.Tensor(
            [
                [1, 0, 3],
                [1, 2, 0],
            ]
        ).long()
        tokens = torch.rand(2, 4, 5)
        values = torch.rand(2, 3, 5)
        new_tokens = utils.set_at_index(tokens, index, values)

        # make sure that values are copied correctly
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                self._assert_tensor_equal(new_tokens[i, index[i, j]], values[i, j])

    def test_mask_at_index(self, seed=0):
        torch.manual_seed(seed)
        index = torch.Tensor(
            [
                [1, 0, 3],
                [1, 2, 0],
            ]
        ).long()
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

        height_patches = height // patch_size
        width_patches = width // patch_size
        num_patches = height_patches * width_patches
        patch_dim = channels * patch_size**2

        self.assertListEqual(
            list(batch_patches.shape), [batch_size, num_patches, patch_dim]
        )

        # make sure that patches are correctly formed
        for image, img_patches in zip(images, batch_patches):
            for i in range(height_patches):
                for j in range(width_patches):
                    # extract patch from original image
                    expected_patch = image[
                        :,
                        i * patch_size : (i + 1) * patch_size,
                        j * patch_size : (j + 1) * patch_size,
                    ]
                    # permute and flatten to match order of patchified images
                    expected_patch = expected_patch.permute(1, 2, 0).flatten()
                    img_patch = img_patches[i * width_patches + j]
                    self._assert_tensor_equal(img_patch, expected_patch)

    def test_unpatchify(self, seed=0):
        torch.manual_seed(seed)
        batch_size, channels, height, width = (2, 3, 8, 8)
        patch_size = 4
        images = torch.rand(batch_size, channels, height, width)
        batch_patches = utils.patchify(images, patch_size)
        unpatched_images = utils.unpatchify(batch_patches, patch_size)

        self._assert_tensor_equal(images, unpatched_images)

    def _test_random_token_mask(
        self, seed=0, mask_ratio=0.6, mask_class_token=False, device="cpu"
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
        expected_idx = (
            torch.arange(seq_length).repeat(batch_size).reshape(batch_size, seq_length)
        )
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
        self._test_random_token_mask_parameters(device="cpu")

    def test_nearest_neighbors(self):
        # Test input with shape (batch_size, map_size_0, num_input_maps)
        input_maps = torch.tensor(
            [
                [[1, 4], [2, 5], [3, 6]],
                [[7, 10], [8, 11], [9, 12]],
                [[13, 16], [14, 17], [15, 18]],
            ]
        )
        print(input_maps.shape)
        # Test candidate maps with shape (batch_size, map_size_1, num_candidate_maps)
        candidate_maps = torch.tensor(
            [
                [[1, 1], [2, 2], [3, 3]],
                [[1, 1], [2, 2], [3, 3]],
                [[1, 1], [2, 2], [3, 3]],
            ]
        )
        print(candidate_maps.shape)
        # Test distances with shape (batch_size, map_size_0, map_size_1)
        distances = torch.tensor(
            [[[0, 1, 2], [1, 0, 3]], [[4, 3, 2], [3, 2, 1]], [[2, 3, 4], [3, 4, 5]]]
        )
        print(input_maps.shape)
        # Test num_matches = 2
        input_maps_filtered, candidate_maps_filtered = nearest_neighbors(
            input_maps, candidate_maps, distances, num_matches=2
        )
        assert input_maps_filtered.shape == (3, 2, 2)
        assert input_maps_filtered.equal(
            torch.tensor([[[1, 4], [2, 5]], [[8, 11], [7, 10]], [[13, 16], [14, 17]]])
        )
        assert candidate_maps_filtered.shape == (3, 2, 2)
        assert candidate_maps_filtered.equal(
            torch.tensor([[[1, 1], [2, 2]], [[3, 3], [3, 3]], [[1, 1], [1, 1]]])
        )
        # Test num_matches = 1
        input_maps_filtered, candidate_maps_filtered = nearest_neighbors(
            input_maps, candidate_maps, distances, num_matches=1
        )
        assert input_maps_filtered.shape == (3, 1, 2)
        assert input_maps_filtered.equal(
            torch.tensor([[[1, 4]], [[8, 11]], [[13, 16]]])
        )
        assert candidate_maps_filtered.shape == (3, 1, 2)
        assert candidate_maps_filtered.equal(
            torch.tensor([[[1, 1]], [[3, 3]], [[1, 1]]])
        )

    @unittest.skipUnless(torch.cuda.is_available(), "No cuda available")
    def test_random_token_mask_cuda(self):
        self._test_random_token_mask_parameters(device="cuda")


@pytest.mark.parametrize(
    "mask_ratio, expected_num_images_masked",
    [
        (0.0, 0),
        (0.4, 2),
        (0.6, 3),
        (1.0, 5),
    ],
)
def test_random_block_mask__mask_ratio(
    mask_ratio: float, expected_num_images_masked: int
) -> None:
    mask = utils.random_block_mask(
        size=(5, 14, 14),
        mask_ratio=mask_ratio,
    )
    num_images_masked = sum(m.sum() > 0 for m in mask)
    assert num_images_masked == expected_num_images_masked


@pytest.mark.parametrize(
    "min_image_mask_ratio, max_image_mask_ratio",
    [(0.0, 0.0), (0.4, 0.6), (1.0, 1.0)],
)
def test_random_block_mask__min_max_image_mask_ratio(
    min_image_mask_ratio: float, max_image_mask_ratio: float
) -> None:
    torch.manual_seed(0)
    random.seed(0)
    mask = utils.random_block_mask(
        size=(5, 14, 14),
        min_image_mask_ratio=min_image_mask_ratio,
        max_image_mask_ratio=max_image_mask_ratio,
        min_num_mask_per_block=0,
    )
    num_masked = mask.sum()
    num_patches = 5 * 14 * 14
    # Divide lower bound by 4 because the bound is not strict as fewer patches than
    # min_image_mask_ratio * num_patches can be masked. This is because there is a
    # limited number of attempts to find a valid mask that satisfies all constraints.
    assert (
        min_image_mask_ratio * num_patches / 4
        <= num_masked
        <= max_image_mask_ratio * num_patches
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_random_block_mask__device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    mask = utils.random_block_mask(size=(2, 14, 14), device=device)
    assert mask.device.type == device


@pytest.mark.parametrize(
    "size,num_mask,min_num_mask_per_block,max_num_mask_per_block",
    [
        ((8, 12), 0, 0, None),
        ((8, 12), 10, 0, None),
        ((8, 12), 10, 4, None),
        ((8, 12), 10, 10, None),
        ((8, 12), 10, 0, 4),
        ((8, 12), 10, 0, 10),
        ((8, 12), 8 * 12, 1, None),
    ],
)
def test_random_block_mask_image__num_mask_per_block(
    size: Tuple[int, int],
    num_mask: int,
    min_num_mask_per_block: int,
    max_num_mask_per_block: Optional[int],
) -> None:
    torch.manual_seed(0)
    random.seed(0)
    mask = utils.random_block_mask_image(
        size=size,
        num_mask=num_mask,
        min_num_mask_per_block=min_num_mask_per_block,
        max_num_mask_per_block=max_num_mask_per_block,
    )
    assert min_num_mask_per_block <= mask.sum() <= num_mask
    assert mask.shape == size


def test_random_block_mask_image__num_mask_per_block__fail() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "max_num_mask_per_block must be greater or equal to min_num_mask_per_block"
        ),
    ):
        utils.random_block_mask_image(
            size=(14, 14),
            num_mask=10,
            min_num_mask_per_block=10,
            max_num_mask_per_block=4,
        )


@pytest.mark.parametrize(
    "min_aspect,max_aspect",
    [
        (0.1, None),
        (0.1, 0.3),
        (0.1, 3.0),
        (0.3, 0.3),
    ],
)
def test_random_block_mask_image__aspect_ratio(
    min_aspect: float, max_aspect: Optional[float]
) -> None:
    mask = utils.random_block_mask_image(
        size=(14, 14), num_mask=10, min_aspect=min_aspect, max_aspect=max_aspect
    )
    assert mask.sum() > 0


def test_random_block_mask_image__aspect_ratio_one() -> None:
    """With aspect ratio 1.0 and num_mask=min_num_mask_per_block we expect a single,
    square masked block."""
    mask = utils.random_block_mask_image(
        size=(14, 14),
        num_mask=9,
        min_num_mask_per_block=9,
        min_aspect=1.0,
        max_aspect=1.0,
    )
    assert mask.sum(dim=0).max() == 3
    assert mask.sum(dim=1).max() == 3


def test_random_block_mask_image__aspect_ratio_fail() -> None:
    with pytest.raises(
        ValueError, match="max_aspect must be greater or equal to min_aspect"
    ):
        utils.random_block_mask_image(
            size=(14, 14),
            num_mask=10,
            min_aspect=3.0,
            max_aspect=1.0,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_random_block_mask_image__device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    mask = utils.random_block_mask_image(
        size=(14, 14),
        num_mask=10,
        device=device,
    )
    assert mask.device.type == device


@pytest.mark.parametrize(
    "x, y, expected",
    [
        # Tests with x, y having shape (1, 2, 2)
        ([[[0, 1], [1, 0]]], [[[0, 1], [1, 0]]], [[0, 1]]),
        ([[[0, 1], [1, 0]]], [[[1, 0], [0, 1]]], [[1, 0]]),
        ([[[0, 1], [1, 0]]], [[[0, -1], [-1, 0]]], [[1, 0]]),
        # Test with x, y having shape (3, 2, 2)
        (
            [
                [[0, 1], [1, 0]],
                [[0, 1], [1, 0]],
                [[0, 1], [1, 0]],
            ],
            [
                [[0, 1], [1, 0]],
                [[1, 0], [0, 1]],
                [[0, -1], [-1, 0]],
            ],
            [
                [0, 1],
                [1, 0],
                [1, 0],
            ],
        ),
    ],
)
def test_most_similar_index(
    x: List[List[List[float]]],
    y: List[List[List[float]]],
    expected: List[List[int]],
) -> None:
    tx = torch.tensor(x, dtype=torch.float)
    ty = torch.tensor(y, dtype=torch.float)
    texpected = torch.tensor(expected)
    result = utils.most_similar_index(tx, ty)
    print(result)  # For easier debugging if test fails.
    assert torch.equal(result, texpected)


@pytest.mark.parametrize(
    "x, y, y_values, expected",
    [
        # Tests with x, y having shape (1, 2, 2) and y_values having shape (1, 2, 3)
        (
            [[[0, 1], [1, 0]]],  # x
            [[[0, 1], [1, 0]]],  # y
            [[[0, 0, 0], [1, 1, 1]]],  # y_values
            [[[0, 0, 0], [1, 1, 1]]],  # expected
        ),
        (
            [[[0, 1], [1, 0]]],  # x
            [[[1, 0], [0, 1]]],  # y
            [[[0, 0, 0], [1, 1, 1]]],  # y_values
            [[[1, 1, 1], [0, 0, 0]]],  # expected
        ),
        (
            [[[0, 1], [1, 0]]],  # x
            [[[0, -1], [-1, 0]]],  # y
            [[[0, 0, 0], [1, 1, 1]]],  # y_values
            [[[1, 1, 1], [0, 0, 0]]],  # expected
        ),
        # Test with x, y having shape (3, 2, 2) and y_values having shape (3, 2, 3)
        (
            [  # x
                [[0, 1], [1, 0]],
                [[0, 1], [1, 0]],
                [[0, 1], [1, 0]],
            ],
            [  # y
                [[0, 1], [1, 0]],
                [[1, 0], [0, 1]],
                [[0, -1], [-1, 0]],
            ],
            [  # y_values
                [[0, 0, 0], [1, 1, 1]],
                [[2, 2, 2], [3, 3, 3]],
                [[4, 4, 4], [5, 5, 5]],
            ],
            [  # expected
                [[0, 0, 0], [1, 1, 1]],
                [[3, 3, 3], [2, 2, 2]],
                [[5, 5, 5], [4, 4, 4]],
            ],
        ),
    ],
)
def test_select_most_similar(
    x: List[List[List[float]]],
    y: List[List[List[float]]],
    y_values: List[List[List[float]]],
    expected: List[List[List[float]]],
) -> None:
    tx = torch.tensor(x, dtype=torch.float)
    ty = torch.tensor(y, dtype=torch.float)
    ty_values = torch.tensor(y_values, dtype=torch.float)
    texpected = torch.tensor(expected, dtype=torch.float)
    result = utils.select_most_similar(tx, ty, ty_values)
    print(result)  # For easier debugging if test fails.
    assert torch.equal(result, texpected)


@pytest.mark.parametrize(
    "seq_length, mask_ratio, mask_class_token, expected_num_masked",
    [
        (5, 0.5, False, 2),
        (5, 0.5, True, 3),
        (257, 0.75, False, 192),  # From issue #1583
        (257, 0.75, True, 193),  # From issue #1583
    ],
)
def test_random_token_mask__mask_class_token(
    seq_length: int, mask_ratio: float, mask_class_token: bool, expected_num_masked: int
) -> None:
    torch.manual_seed(0)
    batch_size = 2
    idx_keep, idx_mask = utils.random_token_mask(
        size=(batch_size, seq_length),
        mask_ratio=mask_ratio,
        mask_class_token=mask_class_token,
    )
    assert idx_mask.shape == (batch_size, expected_num_masked)
    assert idx_keep.shape == (batch_size, seq_length - expected_num_masked)


def test_get_weight_decay_parameters() -> None:
    linear = nn.Linear(10, 10)
    batch_norm1d = nn.BatchNorm1d(10)
    conv = nn.Conv2d(3, 3, 3)
    batch_norm2d = nn.BatchNorm2d(3)
    layer_norm = nn.LayerNorm(10)
    sequential = nn.Sequential(linear, batch_norm1d, conv, batch_norm2d, layer_norm)
    params, params_no_weight_decay = utils.get_weight_decay_parameters(
        modules=[sequential]
    )
    assert len(params) == 2
    assert len(params_no_weight_decay) == 8
    assert params[0] is linear.weight
    assert params[1] is conv.weight
    assert params_no_weight_decay[0] is linear.bias
    assert params_no_weight_decay[1] is batch_norm1d.weight
    assert params_no_weight_decay[2] is batch_norm1d.bias
    assert params_no_weight_decay[3] is conv.bias
    assert params_no_weight_decay[4] is batch_norm2d.weight
    assert params_no_weight_decay[5] is batch_norm2d.bias
    assert params_no_weight_decay[6] is layer_norm.weight
    assert params_no_weight_decay[7] is layer_norm.bias


def test_get_weight_decay_parameters__nested() -> None:
    linear = nn.Linear(10, 10)
    batch_norm1d = nn.BatchNorm1d(10)
    sequential = nn.Sequential(
        nn.Sequential(linear, batch_norm1d),
    )
    params, params_no_weight_decay = utils.get_weight_decay_parameters(
        modules=[sequential]
    )
    assert len(params) == 1
    assert len(params_no_weight_decay) == 3
    assert params[0] is linear.weight
    assert params_no_weight_decay[0] is linear.bias
    assert params_no_weight_decay[1] is batch_norm1d.weight
    assert params_no_weight_decay[2] is batch_norm1d.bias


def test_get_weight_decay_parameters__batch_norm() -> None:
    bn1d = nn.BatchNorm1d(10)
    bn2d = nn.BatchNorm2d(10)
    params, params_no_weight_decay = utils.get_weight_decay_parameters(
        modules=[bn1d, bn2d], decay_norm=True
    )
    assert len(params) == 4
    assert len(params_no_weight_decay) == 0
    assert params[0] is bn1d.weight
    assert params[1] is bn1d.bias
    assert params[2] is bn2d.weight
    assert params[3] is bn2d.bias


def test_get_weight_decay_parameters__no_batch_norm() -> None:
    bn1d = nn.BatchNorm1d(10)
    bn2d = nn.BatchNorm2d(10)
    params, params_no_weight_decay = utils.get_weight_decay_parameters(
        modules=[bn1d, bn2d], decay_norm=False
    )
    print(params, params_no_weight_decay)
    assert len(params) == 0
    assert len(params_no_weight_decay) == 4
    assert params_no_weight_decay[0] is bn1d.weight
    assert params_no_weight_decay[1] is bn1d.bias
    assert params_no_weight_decay[2] is bn2d.weight
    assert params_no_weight_decay[3] is bn2d.bias


def test_get_weight_decay_parameters__bias() -> None:
    linear = nn.Linear(10, 10)
    param, param_no_weight_decay = utils.get_weight_decay_parameters(
        modules=[linear], decay_bias=True
    )
    assert len(param) == 2
    assert len(param_no_weight_decay) == 0
    assert param[0] is linear.weight
    assert param[1] is linear.bias


def test_get_weight_decay_parameters__no_bias() -> None:
    linear = nn.Linear(10, 10)
    param, param_no_weight_decay = utils.get_weight_decay_parameters(
        modules=[linear], decay_bias=False
    )
    assert len(param) == 1
    assert len(param_no_weight_decay) == 1
    assert param[0] is linear.weight
    assert param_no_weight_decay[0] is linear.bias


def test_get_named_leaf_modules() -> None:
    linear1 = nn.Linear(10, 10)
    linear2 = nn.Linear(10, 10)
    sequential1 = nn.Sequential(linear1, linear2)
    sequential2 = nn.Sequential(sequential1)
    assert utils.get_named_leaf_modules(linear1) == {"": linear1}
    assert utils.get_named_leaf_modules(sequential1) == {"0": linear1, "1": linear2}
    assert utils.get_named_leaf_modules(sequential2) == {"0.0": linear1, "0.1": linear2}


def test_normalize_mean_var() -> None:
    x = torch.tensor([1.0, 2.0, 3.0])
    norm = utils.normalize_mean_var(x).tolist()
    assert norm[0] == pytest.approx(-1)
    assert norm[1] == pytest.approx(0.0)
    assert norm[2] == pytest.approx(1)

    x = torch.rand(2, 3, 4)
    norm = utils.normalize_mean_var(x)
    assert torch.allclose(norm.mean(dim=-1), torch.tensor(0.0), rtol=0.0001, atol=1e-5)
    assert torch.allclose(norm.var(dim=-1), torch.tensor(1.0), rtol=0.0001, atol=1e-5)


def test_update_drop_path_rate__uniform() -> None:
    pytest.importorskip("timm.models.vision_transformer")
    from timm.layers import DropPath
    from timm.models.vision_transformer import VisionTransformer

    model = VisionTransformer(drop_path_rate=0.2, depth=4)
    utils.update_drop_path_rate(model=model, drop_path_rate=0.1, mode="uniform")

    for drop_path in [
        model.blocks[0].drop_path1,
        model.blocks[0].drop_path2,
        model.blocks[-1].drop_path1,
        model.blocks[-1].drop_path2,
    ]:
        assert isinstance(drop_path, DropPath)
        assert drop_path.drop_prob == 0.1


def test_update_drop_path_rate__linear() -> None:
    pytest.importorskip("timm.models.vision_transformer")
    from timm.layers import DropPath
    from timm.models.vision_transformer import VisionTransformer

    model = VisionTransformer(drop_path_rate=0, depth=4)
    utils.update_drop_path_rate(model=model, drop_path_rate=0.1, mode="linear")

    for drop_path in [
        model.blocks[0].drop_path1,
        model.blocks[0].drop_path2,
    ]:
        assert isinstance(drop_path, Identity)

    for drop_path in [
        model.blocks[-1].drop_path1,
        model.blocks[-1].drop_path2,
    ]:
        assert isinstance(drop_path, DropPath)
        assert drop_path.drop_prob == 0.1


def test_update_drop_path_rate__unknown_mode() -> None:
    pytest.importorskip("timm.models.vision_transformer")
    from timm.models.vision_transformer import VisionTransformer

    model = VisionTransformer(drop_path_rate=0, depth=4)
    with pytest.raises(ValueError, match="Unknown mode"):
        utils.update_drop_path_rate(model=model, drop_path_rate=0.1, mode="unknown")
