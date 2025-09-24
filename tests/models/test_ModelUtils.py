from __future__ import annotations

import copy
import random
import unittest
from typing import Optional

import pytest
import torch
import torch.nn as nn
from pytest_mock import MockerFixture
from torch import Tensor
from torch.nn import Identity, Parameter

from lightly.models import utils
from lightly.models.utils import (
    _mask_reduce,
    _mask_reduce_batched,
    _no_grad_trunc_normal,
    activate_requires_grad,
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    nearest_neighbors,
    normalize_weight,
    pool_masked,
    update_momentum,
)

is_scatter_reduce_available = hasattr(Tensor, "scatter_reduce_")


@pytest.mark.skipif(
    not is_scatter_reduce_available,
    reason="scatter operations require torch >= 1.12.0",
)
class TestMaskReduce:
    # Type ignore because untyped decorator makes function untyped.
    @pytest.fixture()  # type: ignore[misc]
    def mask1(self) -> Tensor:
        return torch.tensor([[0, 0], [1, 2]], dtype=torch.int64)

    # Type ignore because untyped decorator makes function untyped.
    @pytest.fixture()  # type: ignore[misc]
    def mask2(self) -> Tensor:
        return torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)

    # Type ignore because untyped decorator makes function untyped.
    @pytest.fixture()  # type: ignore[misc]
    def feature_map1(self) -> Tensor:
        feature_map = torch.tensor(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
            dtype=torch.float32,
        )  # (C H W) = (3, 2, 2)
        return feature_map

    # Type ignore because untyped decorator makes function untyped.
    @pytest.fixture()  # type: ignore[misc]
    def feature_map2(self) -> Tensor:
        feature_map = torch.tensor(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
            dtype=torch.float32,
        )  # (C H W) = (3, 2, 2)
        return feature_map

    # Type ignore because untyped decorator makes function untyped.
    @pytest.fixture()  # type: ignore[misc]
    def expected_result1(self) -> Tensor:
        res = torch.tensor(
            [[0.5, 2.0, 3.0], [4.5, 6.0, 7.0], [8.5, 10.0, 11.0]], dtype=torch.float32
        )
        return res

    # Type ignore because untyped decorator makes function untyped.
    @pytest.fixture()  # type: ignore[misc]
    def expected_result2(self) -> Tensor:
        res = torch.tensor(
            [[2.5, 2.5, 0.0], [6.5, 6.5, 0.0], [10.5, 10.5, 0.0]], dtype=torch.float32
        )
        return res

    def test__mask_reduce_batched(
        self,
        feature_map1: Tensor,
        feature_map2: Tensor,
        mask1: Tensor,
        mask2: Tensor,
        expected_result1: Tensor,
        expected_result2: Tensor,
    ) -> None:
        feature_map = torch.stack([feature_map1, feature_map2], dim=0)
        mask = torch.stack([mask1, mask2], dim=0)
        expected_result = torch.stack([expected_result1, expected_result2], dim=0)

        out = _mask_reduce_batched(feature_map, mask, num_cls=3)
        assert (out == expected_result).all()

    def test_masked_pooling_manual(
        self, feature_map2: Tensor, mask2: Tensor, expected_result2: Tensor
    ) -> None:
        out_manual = pool_masked(
            feature_map2.unsqueeze(0), mask2.unsqueeze(0), num_cls=2
        )
        assert out_manual.shape == (1, 3, 2)
        assert (out_manual == expected_result2[:, :2]).all()

    # Type ignore because untyped decorator makes function untyped.
    @pytest.mark.parametrize(
        "feature_map, mask, expected_result",
        [
            (
                torch.tensor(
                    [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
                    dtype=torch.float32,
                ),
                torch.tensor([[0, 0], [1, 2]], dtype=torch.int64),
                torch.tensor(
                    [[0.5, 2.0, 3.0], [4.5, 6.0, 7.0], [8.5, 10.0, 11.0]],
                    dtype=torch.float32,
                ),
            ),
            (
                torch.tensor(
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
                    dtype=torch.float32,
                ),
                torch.tensor([[1, 0], [0, 1]], dtype=torch.int64),
                torch.tensor(
                    [[2.5, 2.5, 0.0], [6.5, 6.5, 0.0], [10.5, 10.5, 0.0]],
                    dtype=torch.float32,
                ),
            ),
        ],
    )  # type: ignore[misc]
    def test__mask_reduce(
        self, feature_map: Tensor, mask: Tensor, expected_result: Tensor
    ) -> None:
        out = _mask_reduce(feature_map, mask, num_cls=3)
        assert (out == expected_result).all()

    def test_singular_mask(self) -> None:
        b, c, h, w = 4, 16, 4, 4
        proj = torch.randn((b, c, h, w))
        mask = torch.zeros((b, h, w), dtype=torch.int64)
        pooled_global = torch.mean(proj, dim=(2, 3)).unsqueeze(-1)  # (b, c, 1=num_cls)
        pooled_mask = pool_masked(proj, mask, num_cls=1)  # (b, c, 1=num_cls)
        assert torch.allclose(pooled_global, pooled_mask)


def has_grad(model: nn.Module) -> bool:
    """Helper method to check if a model has `requires_grad` set to True"""
    has_grad_ = False
    for param in model.parameters():
        if param.requires_grad == True:
            has_grad_ = True
            break
    return has_grad_


class TestModelUtils(unittest.TestCase):
    def _assert_tensor_equal(self, x: Tensor, y: Tensor) -> None:
        # If the assertion fails then only an "assertion is not True" error is
        # shown without showing the contents of x and y. To help debugging, x
        # and y are printed. Note that the output is only shown if the assertion
        # fails.
        print(x)
        print(y)
        self.assertTrue(torch.equal(x, y))

    def test_batch_shuffle(self, seed: int = 0) -> None:
        torch.manual_seed(seed)
        x1 = torch.rand((4, 3, 64, 64))
        x1_shuffled, shuffle = batch_shuffle(x1)
        out1 = batch_unshuffle(x1_shuffled, shuffle)
        self.assertTrue(torch.equal(x1, out1))
        self.assertFalse(torch.equal(x1, x1_shuffled))

    def test_activate_requires_grad(self) -> None:
        model = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.assertTrue(has_grad(model))
        deactivate_requires_grad(model)
        self.assertFalse(has_grad(model))
        activate_requires_grad(model)
        self.assertTrue(has_grad(model))

    def test_momentum_works(self) -> None:
        model = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        model_momentum = copy.deepcopy(model)
        update_momentum(model, model_momentum, 0.99)

    def test_normalize_weight_linear(self) -> None:
        input_dim = 32
        output_dim = 64
        linear = nn.Linear(input_dim, output_dim, bias=False)
        normalize_weight(linear.weight, dim=0)
        self.assertEqual(linear.weight.norm(dim=0).sum(), input_dim)
        normalize_weight(linear.weight, dim=1)
        self.assertEqual(linear.weight.norm(dim=1).sum(), output_dim)

    def test_no_grad_trunc_normal(self, device: str = "cpu", seed: int = 0) -> None:
        torch.manual_seed(seed)
        tensor = torch.rand((8, 16)).to(device)
        a = -2
        b = 2
        _no_grad_trunc_normal(tensor, mean=0, std=1, a=-2, b=2)
        self.assertTrue(tensor.min() >= a)
        self.assertTrue(tensor.max() <= b)

    @unittest.skipUnless(torch.cuda.is_available(), "No cuda available")
    def test_no_grad_trunc_normal_cuda(self) -> None:
        self.test_no_grad_trunc_normal(device="cuda")

    def test_repeat_token(self) -> None:
        token = torch.Tensor([[[1, 2, 3, 4]]])
        out = utils.repeat_token(token, size=(2, 3))
        self.assertEqual(tuple(out.shape), (2, 3, 4))
        self.assertListEqual(out[-1][-1].tolist(), [1, 2, 3, 4])

    def test_expand_index_like(self, seed: int = 0) -> None:
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

    def test_get_at_index(self, seed: int = 0) -> None:
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

    def test_set_at_index(self, seed: int = 0) -> None:
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

    def test_mask_at_index(self, seed: int = 0) -> None:
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

    def test_prepend_class_token(self, seed: int = 0) -> None:
        torch.manual_seed(seed)
        tokens = torch.rand(2, 3, 5)
        class_token = torch.rand(1, 1, 5)
        new_tokens = utils.prepend_class_token(tokens, class_token)
        self.assertListEqual(list(new_tokens.shape), [2, 4, 5])

        # make sure that class token is inserted in correct place
        for i in range(new_tokens.shape[0]):
            self._assert_tensor_equal(new_tokens[i][0], class_token[0, 0])

    def test_patchify(self, seed: int = 0) -> None:
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

    def test_unpatchify(self, seed: int = 0) -> None:
        torch.manual_seed(seed)
        batch_size, channels, height, width = (2, 3, 8, 8)
        patch_size = 4
        images = torch.rand(batch_size, channels, height, width)
        batch_patches = utils.patchify(images, patch_size)
        unpatched_images = utils.unpatchify(batch_patches, patch_size)

        self._assert_tensor_equal(images, unpatched_images)

    def _test_random_token_mask(
        self,
        seed: int = 0,
        mask_ratio: float = 0.6,
        mask_class_token: bool = False,
        device: str = "cpu",
    ) -> None:
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

    def _test_random_token_mask_parameters(self, device: str) -> None:
        for mask_ratio in [0, 0.6, 1.0]:
            for mask_class_token in [False, True]:
                self._test_random_token_mask(
                    mask_ratio=mask_ratio,
                    mask_class_token=mask_class_token,
                    device=device,
                )

    def test_random_token_mask(self) -> None:
        self._test_random_token_mask_parameters(device="cpu")

    def test_nearest_neighbors(self) -> None:
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
    def test_random_token_mask_cuda(self) -> None:
        self._test_random_token_mask_parameters(device="cuda")


# Type ignore because untyped decorator makes function untyped.
@pytest.mark.parametrize(
    "mask, expected",
    [
        (
            [
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [[2, 2], [2, 2], [2, 2]],
                [[2, 2], [2, 2], [2, 2]],
            ],
        ),
        (
            [
                [1, 1, 1],
                [1, 1, 1],
            ],
            [
                [[3, 3], [3, 3], [3, 3]],
                [[3, 3], [3, 3], [3, 3]],
            ],
        ),
        (
            [
                [0, 1, 0],
                [1, 0, 1],
            ],
            [
                [[2, 2], [3, 3], [2, 2]],
                [[3, 3], [2, 2], [3, 3]],
            ],
        ),
    ],
)  # type: ignore[misc]
def test_mask_bool(mask: Tensor, expected: Tensor) -> None:
    tokens = torch.zeros(2, 3, 2) + 2
    mask_token = torch.zeros(1, 1, 2) + 3
    result = utils.mask_bool(
        tokens=tokens, mask=torch.tensor(mask, dtype=torch.bool), mask_token=mask_token
    )
    assert torch.allclose(result, torch.tensor(expected, dtype=torch.float))


# Type ignore because untyped decorator makes function untyped.
@pytest.mark.parametrize(
    "mask_ratio, expected_num_images_masked",
    [
        (0.0, 0),
        (0.4, 2),
        (0.6, 3),
        (1.0, 5),
    ],
)  # type: ignore[misc]
def test_random_block_mask__mask_ratio(
    mask_ratio: float, expected_num_images_masked: int
) -> None:
    mask = utils.random_block_mask(
        size=(5, 14, 14),
        batch_mask_ratio=mask_ratio,
    )
    num_images_masked = sum(m.sum() > 0 for m in mask)
    assert num_images_masked == expected_num_images_masked


# Type ignore because untyped decorator makes function untyped.
@pytest.mark.parametrize(
    "min_image_mask_ratio, max_image_mask_ratio",
    [(0.0, 0.0), (0.4, 0.6), (1.0, 1.0)],
)  # type: ignore[misc]
def test_random_block_mask__min_max_image_mask_ratio(
    min_image_mask_ratio: float, max_image_mask_ratio: float
) -> None:
    torch.manual_seed(0)
    random.seed(0)
    mask = utils.random_block_mask(
        size=(5, 14, 14),
        min_image_mask_ratio=min_image_mask_ratio,
        max_image_mask_ratio=max_image_mask_ratio,
        min_num_masks_per_block=0,
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


# Type ignore because untyped decorator makes function untyped.
@pytest.mark.parametrize("device", ["cpu", "cuda"])  # type: ignore[misc]
def test_random_block_mask__device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    mask = utils.random_block_mask(size=(2, 14, 14), device=device)
    assert mask.device.type == device


# Type ignore because untyped decorator makes function untyped.
@pytest.mark.parametrize(
    "size,num_masks,min_num_masks_per_block,max_num_masks_per_block",
    [
        ((8, 12), 0, 0, None),
        ((8, 12), 10, 0, None),
        ((8, 12), 10, 4, None),
        ((8, 12), 10, 10, None),
        ((8, 12), 10, 0, 4),
        ((8, 12), 10, 0, 10),
        ((8, 12), 8 * 12, 1, None),
    ],
)  # type: ignore[misc]
def test_random_block_mask_image__num_mask_per_block(
    size: tuple[int, int],
    num_masks: int,
    min_num_masks_per_block: int,
    max_num_masks_per_block: int | None,
) -> None:
    torch.manual_seed(0)
    random.seed(0)
    mask = utils.random_block_mask_image(
        size=size,
        num_masks=num_masks,
        min_num_masks_per_block=min_num_masks_per_block,
        max_num_masks_per_block=max_num_masks_per_block,
    )
    assert min_num_masks_per_block <= mask.sum() <= num_masks
    assert mask.shape == size


def test_random_block_mask_image__num_mask_per_block__fail() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "max_num_masks_per_block must be greater or equal to min_num_masks_per_block"
        ),
    ):
        utils.random_block_mask_image(
            size=(14, 14),
            num_masks=10,
            min_num_masks_per_block=10,
            max_num_masks_per_block=4,
        )


# Type ignore because untyped decorator makes function untyped.
@pytest.mark.parametrize(
    "min_block_aspect_ratio,max_block_aspect_ratio",
    [
        (0.1, None),
        (0.1, 0.3),
        (0.1, 3.0),
        (0.3, 0.3),
    ],
)  # type: ignore[misc]
def test_random_block_mask_image__aspect_ratio(
    min_block_aspect_ratio: float, max_block_aspect_ratio: float | None
) -> None:
    mask = utils.random_block_mask_image(
        size=(14, 14),
        num_masks=10,
        min_block_aspect_ratio=min_block_aspect_ratio,
        max_block_aspect_ratio=max_block_aspect_ratio,
    )
    assert mask.sum() > 0


def test_random_block_mask_image__aspect_ratio_one() -> None:
    """With aspect ratio 1.0 and num_mask=min_num_masks_per_block we expect a single,
    square masked block."""
    mask = utils.random_block_mask_image(
        size=(14, 14),
        num_masks=9,
        min_num_masks_per_block=9,
        min_block_aspect_ratio=1.0,
        max_block_aspect_ratio=1.0,
    )
    assert mask.sum(dim=0).max() == 3
    assert mask.sum(dim=1).max() == 3


def test_random_block_mask_image__aspect_ratio_fail() -> None:
    with pytest.raises(
        ValueError,
        match="max_block_aspect_ratio must be greater or equal to min_block_aspect_ratio",
    ):
        utils.random_block_mask_image(
            size=(14, 14),
            num_masks=10,
            min_block_aspect_ratio=3.0,
            max_block_aspect_ratio=1.0,
        )


# Type ignore because untyped decorator makes function untyped.
@pytest.mark.parametrize("device", ["cpu", "cuda"])  # type: ignore[misc]
def test_random_block_mask_image__device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    mask = utils.random_block_mask_image(
        size=(14, 14),
        num_masks=10,
        device=device,
    )
    assert mask.device.type == device


# Type ignore because untyped decorator makes function untyped.
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
)  # type: ignore[misc]
def test_most_similar_index(
    x: list[list[list[float]]],
    y: list[list[list[float]]],
    expected: list[list[int]],
) -> None:
    tx = torch.tensor(x, dtype=torch.float)
    ty = torch.tensor(y, dtype=torch.float)
    texpected = torch.tensor(expected)
    result = utils.most_similar_index(tx, ty)
    print(result)  # For easier debugging if test fails.
    assert torch.equal(result, texpected)


# Type ignore because untyped decorator makes function untyped.
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
)  # type: ignore[misc]
def test_select_most_similar(
    x: list[list[list[float]]],
    y: list[list[list[float]]],
    y_values: list[list[list[float]]],
    expected: list[list[list[float]]],
) -> None:
    tx = torch.tensor(x, dtype=torch.float)
    ty = torch.tensor(y, dtype=torch.float)
    ty_values = torch.tensor(y_values, dtype=torch.float)
    texpected = torch.tensor(expected, dtype=torch.float)
    result = utils.select_most_similar(tx, ty, ty_values)
    print(result)  # For easier debugging if test fails.
    assert torch.equal(result, texpected)


# Type ignore because untyped decorator makes function untyped.
@pytest.mark.parametrize(
    "seq_length, mask_ratio, mask_class_token, expected_num_masked",
    [
        (5, 0.5, False, 2),
        (5, 0.5, True, 3),
        (257, 0.75, False, 192),  # From issue #1583
        (257, 0.75, True, 193),  # From issue #1583
    ],
)  # type: ignore[misc]
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


# Type ignore because untyped decorator makes function untyped.
@pytest.mark.parametrize(
    "strategy, expected_fn",
    [
        ("learn", "initialize_learnable_positional_embedding"),
        ("sincos", "initialize_2d_sine_cosine_positional_embedding"),
        ("skip", None),
    ],
)  # type: ignore[misc]
def test_initialize_positional_embedding(
    strategy: str, expected_fn: Optional[str], mocker: MockerFixture
) -> None:
    if expected_fn is not None:
        mock_fn = mocker.spy(utils, expected_fn)
    pos_embedding = Parameter(torch.rand(1, 1, 64))
    utils.initialize_positional_embedding(
        pos_embedding=pos_embedding, strategy=strategy, num_prefix_tokens=1
    )
    if expected_fn is not None:
        mock_fn.assert_called_once()


def test_initialize_learnable_positional_embedding() -> None:
    pos_embedding = Parameter(torch.ones(1, 1, 64))
    orig_pos_embedding = pos_embedding.clone()
    utils.initialize_learnable_positional_embedding(pos_embedding)
    # Embedding must be learnable.
    assert pos_embedding.requires_grad
    # Embedding must have changed.
    assert torch.any(pos_embedding != orig_pos_embedding)


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
