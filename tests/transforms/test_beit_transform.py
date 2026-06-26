import unittest

import torch

from lightly.transforms import BEITTransform


class TestBEITTransform(unittest.TestCase):
    _GRID = 14
    _N = 14 * 14
    _MASK_RATIO = 0.4

    def _make_transform(self, **kwargs) -> BEITTransform:
        return BEITTransform(**kwargs)

    def test_mask_shape(self) -> None:
        transform = self._make_transform()
        for batch_size in [1, 4, 8]:
            mask = transform.mask_generator(batch_size=batch_size)
            self.assertEqual(mask.shape, (batch_size, self._N))

    def test_mask_dtype_is_bool(self) -> None:
        transform = self._make_transform()
        mask = transform.mask_generator(batch_size=2)
        self.assertEqual(mask.dtype, torch.bool)

    def test_mask_values_are_binary(self) -> None:
        transform = self._make_transform()
        mask = transform.mask_generator(batch_size=4)
        unique = mask.unique()
        self.assertTrue(set(unique.tolist()).issubset({False, True}))

    def test_masking_ratio_at_least_target(self) -> None:
        transform = self._make_transform()
        target = int(self._MASK_RATIO * self._N)
        for _ in range(10):
            mask = transform.mask_generator(batch_size=1)
            self.assertGreaterEqual(mask[0].sum().item(), target)

    def test_masking_ratio_does_not_exceed_all_patches(self) -> None:
        transform = self._make_transform()
        for _ in range(10):
            mask = transform.mask_generator(batch_size=1)
            self.assertLessEqual(mask[0].sum().item(), self._N)

    def test_masking_ratio_respected_across_batch(self) -> None:
        transform = self._make_transform()
        mask = transform.mask_generator(batch_size=8)
        target = int(self._MASK_RATIO * self._N)
        for i in range(8):
            self.assertGreaterEqual(mask[i].sum().item(), target)

    def test_masks_differ_across_samples_in_batch(self) -> None:
        transform = self._make_transform()
        mask = transform.mask_generator(batch_size=4)
        identical = all(torch.equal(mask[0], mask[i]) for i in range(1, 4))
        self.assertFalse(identical)

    def test_masks_differ_across_calls(self) -> None:
        transform = self._make_transform()
        mask_a = transform.mask_generator(batch_size=1)
        mask_b = transform.mask_generator(batch_size=1)
        self.assertFalse(torch.equal(mask_a, mask_b))

    def test_custom_grid_size(self) -> None:
        transform = self._make_transform(input_size=112, patch_size=16)
        mask = transform.mask_generator(batch_size=2)
        self.assertEqual(mask.shape, (2, 49))

    def test_custom_mask_ratio(self) -> None:
        ratio = 0.6
        transform = self._make_transform(mask_ratio=ratio)
        target = int(ratio * self._N)
        mask = transform.mask_generator(batch_size=4)
        for i in range(4):
            self.assertGreaterEqual(mask[i].sum().item(), target)

    def test_min_block_patches_of_one_still_produces_mask(self) -> None:
        transform = self._make_transform(min_block_patches=1)
        mask = transform.mask_generator(batch_size=4)
        for i in range(4):
            self.assertGreater(mask[i].sum().item(), 0)
